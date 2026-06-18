from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax_md import partition, space

jax.config.update("jax_default_matmul_precision", "highest")

HARTREE_TO_KJMOL = 2625.4996382852164

MACE_MODEL_PATHS = {
    "mace-off-s(23)": Path(__file__).resolve().parent / "mace-off-s(23).eqx",
    "mace-off-m(24)": Path(__file__).resolve().parent / "mace-off-m(24).eqx",
}
MACE_MODEL_NAMES = tuple(MACE_MODEL_PATHS)


def get_neighbors(
    positions,
    box=None,
    *,
    cutoff: float,
    cell_atom_threshold: int = 64,
    cell_capacity_multiplier: float = 2.0,
    extra_capacity: int = 0,
    neighbors=None,
    periodic: bool = False,
    dr_threshold: float = 0.0,
):
    num_atoms = int(positions.shape[0])
    use_cell_list = periodic and num_atoms >= cell_atom_threshold
    if periodic:
        if box is None:
            raise ValueError("periodic neighbor lists require a box.")
        jax_box = jnp.swapaxes(jnp.asarray(box, dtype=positions.dtype), -1, -2)
        displacement, _ = space.periodic_general(
            jax_box,
            fractional_coordinates=True,
        )
        neighbor_kwargs = {"box": jax_box}
    else:
        displacement, _ = space.free()
        neighbor_kwargs = {}

    if neighbors is not None:
        neighbors = neighbors.update(positions, **neighbor_kwargs)
    else:
        neighbor_fn = partition.neighbor_list(
            displacement,
            jnp.asarray(1.0, dtype=positions.dtype),
            float(cutoff),
            dr_threshold=float(dr_threshold),
            capacity_multiplier=float(cell_capacity_multiplier),
            disable_cell_list=not use_cell_list,
            mask_self=True,
            fractional_coordinates=periodic,
            format=partition.NeighborListFormat.Dense,
        )
        neighbors = neighbor_fn.allocate(
            positions,
            extra_capacity=int(extra_capacity),
            **neighbor_kwargs,
        )
    return neighbors


def dense_neighbor_edges(
    positions,
    neighbors,
    *,
    box_vectors=None,
    cutoff: float | None = None,
):
    num_atoms = positions.shape[0]
    atom_ids = jnp.arange(num_atoms, dtype=jnp.int32)
    neighbors = jnp.asarray(neighbors, dtype=jnp.int32)
    valid = (neighbors >= 0) & (neighbors < num_atoms)
    safe_neighbors = jnp.where(valid, neighbors, atom_ids[:, None])

    neighbor_positions = positions[safe_neighbors]
    if box_vectors is None:
        edge_vectors = neighbor_positions - positions[:, None, :]
    else:
        displacement, _ = space.periodic_general(
            jnp.swapaxes(jnp.asarray(box_vectors, dtype=positions.dtype), -1, -2),
            fractional_coordinates=True,
        )
        edge_vectors = space.map_neighbor(displacement)(positions, neighbor_positions)

    distances = safe_norm(edge_vectors, axis=-1)
    edge_mask = valid & (safe_neighbors != atom_ids[:, None]) & (distances > 1.0e-8)
    if cutoff is not None:
        edge_mask = edge_mask & (distances < cutoff)
    edge_vectors = jnp.where(edge_mask[..., None], edge_vectors, 0.0)
    return edge_vectors, safe_neighbors, edge_mask


def safe_norm(x: Array, *, axis=-1, keepdims: bool = False, eps: float = 1.0e-24) -> Array:
    return jnp.sqrt(jnp.maximum(jnp.sum(x * x, axis=axis, keepdims=keepdims), eps))


def normalized_silu(x: Array, normalization: float) -> Array:
    return jax.nn.silu(x) / normalization


def polynomial_cutoff(r: Array, r_max: float, p: int = 5) -> Array:
    u = r / r_max
    up = jnp.power(u, p)
    envelope = (
        1.0
        - 0.5 * (p + 1) * (p + 2) * up
        + p * (p + 2) * up * u
        - 0.5 * p * (p + 1) * up * u * u
    )
    return jnp.where(r < r_max, envelope, 0.0)


def bessel_basis(r: Array, r_max: float, num_basis: int) -> Array:
    ns = jnp.arange(1, num_basis + 1, dtype=r.dtype)
    return (
        jnp.sqrt(2.0 / r_max)
        * jnp.pi
        * ns
        / r_max
        * jnp.sinc(ns * r[..., None] / r_max)
    )

def spherical_harmonics_0_to_3(
    edge_vectors: Array,
    coeffs: Array,
    monomials: tuple[tuple[int, int, int], ...],
) -> Array:
    norms = safe_norm(edge_vectors, axis=-1, keepdims=True)
    xyz = edge_vectors / norms
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    cols = []
    for px, py, pz in monomials:
        value = jnp.ones_like(x)
        if px:
            value = value * x**px
        if py:
            value = value * y**py
        if pz:
            value = value * z**pz
        cols.append(value)
    return jnp.stack(cols, axis=-1) @ coeffs.T


class Linear(eqx.Module):
    w: Array
    in_dim: int = eqx.field(static=True)

    def __init__(self, in_dim: int, out_dim: int, *, dtype: Any = jnp.float32, key: Array):
        self.in_dim = in_dim
        self.w = jax.random.normal(key, (in_dim, out_dim), dtype)

    def __call__(self, x: Array) -> Array:
        scale = jnp.sqrt(1.0 / self.in_dim)
        if x.shape[-1] == self.in_dim:
            return scale * jnp.matmul(x, self.w)
        if x.ndim == 3 and x.shape[1] == self.in_dim:
            return scale * jnp.einsum("nci,co->noi", x, self.w)
        raise ValueError(f"Expected feature axis of size {self.in_dim}, got shape {x.shape}.")


class SpeciesLinear(eqx.Module):
    w: Array
    num_species: int = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)

    def __init__(self, num_species: int, in_dim: int, out_dim: int, *, dtype: Any = jnp.float32, key: Array):
        self.num_species = num_species
        self.in_dim = in_dim
        self.w = jax.random.normal(key, (num_species, in_dim, out_dim), dtype)

    def __call__(self, x: Array, species: Array) -> Array:
        scale = jnp.sqrt(1.0 / (self.in_dim * self.num_species))
        weights = self.w[species]
        if x.shape[-1] == self.in_dim:
            return scale * jnp.einsum("n...i,nio->n...o", x, weights)
        if x.ndim == 3 and x.shape[1] == self.in_dim:
            return scale * jnp.einsum("nci,nco->noi", x, weights)
        raise ValueError(f"Expected feature axis of size {self.in_dim}, got shape {x.shape}.")

class MLP(eqx.Module):
    linears: list[Array]
    silu_normalization: float = eqx.field(static=True)
    layer_sizes: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, layer_sizes, *, silu_normalization: float, dtype=jnp.float32, key):
        self.silu_normalization = silu_normalization
        self.layer_sizes = tuple(layer_sizes)
        keys = jax.random.split(key, len(self.layer_sizes) - 1)
        self.linears = [jax.random.normal(k, (i, o), dtype) for k, i, o in zip(keys, self.layer_sizes[:-1], self.layer_sizes[1:])]

    def __call__(self, x: Array) -> Array:
        for i, w in enumerate(self.linears):
            x = jnp.sqrt(1.0 / self.layer_sizes[i]) * jnp.matmul(x, w)
            if i < len(self.linears) - 1:
                x = normalized_silu(x, self.silu_normalization)
        return x


class SymmetricContraction(eqx.Module):
    w0: Array
    w1: Array
    u0_1: Array
    u0_2: Array
    u0_3: Array
    u1_1: Array
    u1_2: Array
    u1_3: Array
    output_vector: bool = eqx.field(static=True)

    def __init__(self, num_species: int, num_features: int, output_vector: bool, *, dtype=jnp.float32, key):
        self.output_vector = output_vector
        self.w0 = jax.random.normal(key, (num_species, 28, num_features), dtype)
        self.w1 = jnp.zeros((num_species, 58, num_features), dtype=dtype)
        self.u0_1 = jnp.zeros((16, 1, 1), dtype=dtype)
        self.u0_2 = jnp.zeros((16, 16, 4, 1), dtype=dtype)
        self.u0_3 = jnp.zeros((16, 16, 16, 23, 1), dtype=dtype)
        self.u1_1 = jnp.zeros((16, 1, 3), dtype=dtype)
        self.u1_2 = jnp.zeros((16, 16, 6, 3), dtype=dtype)
        self.u1_3 = jnp.zeros((16, 16, 16, 51, 3), dtype=dtype)

    def _features(self, x: Array, u1: Array, u2: Array, u3: Array) -> Array:
        phi1 = jnp.einsum("nfa,ami->nfmi", x, u1)
        phi2 = jnp.einsum("nfa,nfb,abmi->nfmi", x, x, u2)
        phi3 = jnp.einsum("nfa,nfb,nfc,abcmi->nfmi", x, x, x, u3)
        return jnp.concatenate([phi1, phi2, phi3], axis=2)

    def __call__(self, blocks: tuple[Array, Array, Array, Array], species: Array) -> tuple[Array, Array | None]:
        x = jnp.concatenate(blocks, axis=-1)
        f0 = self._features(x, self.u0_1, self.u0_2, self.u0_3)
        w0 = jnp.swapaxes(self.w0[species], 1, 2)[..., None]
        out0 = jnp.sum(f0 * w0, axis=2)[..., 0]
        out1 = None
        if self.output_vector:
            f1 = self._features(x, self.u1_1, self.u1_2, self.u1_3)
            w1 = jnp.swapaxes(self.w1[species], 1, 2)[..., None]
            out1 = jnp.sum(f1 * w1, axis=2)
        return out0, out1


class FullMACELayer(eqx.Module):
    linear_up0: Linear
    linear_up1: Linear
    radial_mlp: MLP
    linear_down: list[Linear]
    skip0: SpeciesLinear | None
    linz: list[SpeciesLinear] | None
    sc: SymmetricContraction
    linear_sc0: Linear
    linear_sc1: Linear
    readout_mlp: Linear | None
    readout: Linear
    clebsch_gordan_coefficients: tuple[Array, ...]
    vector_input: bool = eqx.field(static=True)
    vector_output: bool = eqx.field(static=True)
    is_last: bool = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    conv_widths: tuple[int, int, int, int] = eqx.field(static=True)
    sh_dims: tuple[int, ...] = eqx.field(static=True)
    sh_starts: tuple[int, ...] = eqx.field(static=True)
    silu_normalization: float = eqx.field(static=True)

    def __init__(
        self,
        num_features,
        num_species,
        radial_dim,
        epsilon,
        vector_input,
        vector_output,
        has_skip,
        has_linz,
        is_last,
        *,
        sh_dims: tuple[int, ...],
        sh_starts: tuple[int, ...],
        silu_normalization: float,
        dtype=jnp.float32,
        key,
    ):
        keys = iter(jax.random.split(key, 16))
        self.vector_input = vector_input
        self.vector_output = vector_output
        self.is_last = is_last
        self.epsilon = epsilon
        self.sh_dims = sh_dims
        self.sh_starts = sh_starts
        self.silu_normalization = silu_normalization
        self.conv_widths = (2 * num_features, 3 * num_features, 3 * num_features, 2 * num_features) if vector_input else (num_features,) * 4
        self.linear_up0 = Linear(num_features, num_features, dtype=dtype, key=next(keys))
        self.linear_up1 = Linear(num_features, num_features, dtype=dtype, key=next(keys))
        self.radial_mlp = MLP(
            [radial_dim, 64, 64, 64, sum(self.conv_widths)],
            silu_normalization=silu_normalization,
            dtype=dtype,
            key=next(keys),
        )
        self.linear_down = [Linear(w, num_features, dtype=dtype, key=next(keys)) for w in self.conv_widths]
        self.skip0 = SpeciesLinear(num_species, num_features, num_features, dtype=dtype, key=next(keys)) if has_skip else None
        self.linz = [SpeciesLinear(num_species, num_features, num_features, dtype=dtype, key=next(keys)) for _ in range(4)] if has_linz else None
        self.sc = SymmetricContraction(num_species, num_features, vector_output, dtype=dtype, key=next(keys))
        self.linear_sc0 = Linear(num_features, num_features, dtype=dtype, key=next(keys))
        self.linear_sc1 = Linear(num_features, num_features, dtype=dtype, key=next(keys))
        if is_last:
            self.readout_mlp = Linear(num_features, 16, dtype=dtype, key=next(keys))
            self.readout = Linear(16, 1, dtype=dtype, key=next(keys))
        else:
            self.readout_mlp = None
            self.readout = Linear(num_features, 1, dtype=dtype, key=next(keys))
        self.clebsch_gordan_coefficients = (
            jnp.zeros((3, 3, 1), dtype=dtype),
            jnp.zeros((3, 1, 3), dtype=dtype),
            jnp.zeros((3, 5, 3), dtype=dtype),
            jnp.zeros((3, 3, 5), dtype=dtype),
            jnp.zeros((3, 7, 5), dtype=dtype),
            jnp.zeros((3, 5, 7), dtype=dtype),
        )

    def _messages_scalar_input(self, h0: Array, y: Array) -> list[Array]:
        hs = h0[..., :, None]
        out = []
        for l, start in enumerate(self.sh_starts):
            yl = y[..., start:start + self.sh_dims[l]]
            out.append(hs * yl[..., None, :])
        return out

    def _messages_vector_input(self, h0: Array, h1: Array, y: Array) -> list[Array]:
        hs = h0[..., :, None]
        y0 = y[..., 0:1]
        y1 = y[..., 1:4]
        y2 = y[..., 4:9]
        y3 = y[..., 9:16]
        c11_0, c10_1, c12_1, c11_2, c13_2, c12_3 = self.clebsch_gordan_coefficients
        out0 = jnp.concatenate([hs * y0[..., None, :], jnp.einsum("...fi,...j,ijk->...fk", h1, y1, c11_0)], axis=-2)
        out1 = jnp.concatenate([hs * y1[..., None, :], jnp.einsum("...fi,...j,ijk->...fk", h1, y0, c10_1), jnp.einsum("...fi,...j,ijk->...fk", h1, y2, c12_1)], axis=-2)
        out2 = jnp.concatenate([hs * y2[..., None, :], jnp.einsum("...fi,...j,ijk->...fk", h1, y1, c11_2), jnp.einsum("...fi,...j,ijk->...fk", h1, y3, c13_2)], axis=-2)
        out3 = jnp.concatenate([hs * y3[..., None, :], jnp.einsum("...fi,...j,ijk->...fk", h1, y2, c12_3)], axis=-2)
        return [out0, out1, out2, out3]

    def __call__(self, h0: Array, h1: Array | None, species: Array, y: Array, radial: Array, senders: Array, edge_mask: Array) -> tuple[Array, Array | None, Array]:
        skip = self.skip0(h0, species) if self.skip0 is not None else None
        h0u = self.linear_up0(h0)
        if self.vector_input:
            assert h1 is not None
            h1u = self.linear_up1(h1)
            msg = self._messages_vector_input(h0u[senders], h1u[senders], y)
        else:
            msg = self._messages_scalar_input(h0u[senders], y)
        mix = self.radial_mlp(radial)
        pieces = jnp.split(mix, np.cumsum(self.conv_widths)[:-1], axis=-1)
        blocks = []
        for l in range(4):
            weighted = self.epsilon * msg[l] * pieces[l][..., :, None]
            weighted = jnp.where(edge_mask[..., None, None], weighted, 0.0)
            agg = jnp.sum(weighted, axis=1)
            down = self.linear_down[l](agg)
            blocks.append(down)
        if self.linz is not None:
            blocks = [self.linz[l](blocks[l], species) for l in range(4)]
        out0, out1 = self.sc(tuple(blocks), species)
        out0 = self.linear_sc0(out0)
        if self.vector_output:
            assert out1 is not None
            out1 = self.linear_sc1(out1)
        if skip is not None:
            out0 = out0 + skip
        layer_out = out0
        if self.is_last:
            assert self.readout_mlp is not None
            layer_out = normalized_silu(self.readout_mlp(layer_out), self.silu_normalization)
        e = jnp.squeeze(self.readout(layer_out), axis=-1)
        return out0, out1, e


class MACEModel(eqx.Module):
    embedding: Array
    layers: list[FullMACELayer]
    offsets: Array
    sh_coeffs: Array
    name: str = eqx.field(static=True)
    num_species: int = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    num_radial_basis: int = eqx.field(static=True)
    radial_polynomial_p: int = eqx.field(static=True)
    num_features: int = eqx.field(static=True)
    hidden_has_vector: bool = eqx.field(static=True)
    silu_normalization: float = eqx.field(static=True)
    neighbor_cell_atom_threshold: int = eqx.field(static=True)
    neighbor_cell_capacity_multiplier: float = eqx.field(static=True)
    sh_dims: tuple[int, ...] = eqx.field(static=True)
    sh_starts: tuple[int, ...] = eqx.field(static=True)
    sh_monomials: tuple[tuple[int, int, int], ...] = eqx.field(static=True)

    def __init__(self, offsets, config: dict[str, Any], *, dtype=jnp.float32, key):
        self.offsets = jnp.asarray(np.asarray(offsets).reshape(-1), dtype=dtype)
        self.name = str(config["name"])
        self.num_species = int(config["num_species"])
        self.cutoff = float(config["cutoff"])
        self.num_radial_basis = int(config["num_radial_basis"])
        self.radial_polynomial_p = int(config["radial_polynomial_p"])
        self.num_features = int(config["num_features"])
        self.hidden_has_vector = bool(config["hidden_has_vector"])
        self.silu_normalization = float(config["silu_normalization"])
        self.neighbor_cell_atom_threshold = int(config.get("neighbor_cell_atom_threshold", 192))
        self.neighbor_cell_capacity_multiplier = float(
            config.get("neighbor_cell_capacity_multiplier", 1.25)
        )
        self.sh_dims = tuple(int(x) for x in config["sh_dims"])
        self.sh_starts = tuple(int(x) for x in config["sh_starts"])
        self.sh_monomials = tuple(tuple(int(v) for v in row) for row in config["sh_monomials"])
        if int(config["correlation"]) != 3:
            raise ValueError("This MACE implementation requires correlation=3.")
        num_layers = int(config["num_layers"])
        epsilon = 1.0 / float(config["avg_num_neighbors"])
        keys = jax.random.split(key, num_layers + 1)
        self.embedding = jax.random.normal(keys[0], (self.num_species, self.num_features), dtype)
        self.sh_coeffs = jnp.zeros((sum(self.sh_dims), len(self.sh_monomials)), dtype=dtype)
        self.layers = [
            FullMACELayer(
                self.num_features,
                self.num_species,
                self.num_radial_basis,
                epsilon,
                vector_input=(i > 0 and self.hidden_has_vector),
                vector_output=(i == 0 and self.hidden_has_vector),
                has_skip=(i > 0),
                has_linz=(i == 0),
                is_last=(i == num_layers - 1),
                sh_dims=self.sh_dims,
                sh_starts=self.sh_starts,
                silu_normalization=self.silu_normalization,
                dtype=dtype,
                key=keys[i + 1],
            )
            for i in range(num_layers)
        ]

    def node_energies(
        self,
        positions,
        species,
        *,
        neighbor_idx,
        box_vectors=None,
    ) -> Array:
        species = jnp.asarray(species, dtype=jnp.int32)
        
        # Convert from flat edge format to dense packed format
        edge_vectors, senders, edge_mask = dense_neighbor_edges(
            positions,
            neighbor_idx,
            box_vectors=box_vectors,
            cutoff=float(self.cutoff),
        )
        senders = jnp.where(edge_mask, senders, 0)

        h0 = self.embedding[species] / jnp.sqrt(self.num_species)
        h1 = None
        distances = safe_norm(edge_vectors, axis=-1)
        # Borrowed from https://github.com/atomicarchitects/nequix/blob/da0fb241f417dad1afafa4e723ad867667ee7445/nequix/model.py#L385-L386
        radial_basis = bessel_basis(distances, self.cutoff, self.num_radial_basis) * polynomial_cutoff(
            distances,
            self.cutoff,
            self.radial_polynomial_p,
        )[..., None]
        radial_basis = jnp.where(edge_mask[..., None], radial_basis, 0.0)

        sph = spherical_harmonics_0_to_3(edge_vectors, self.sh_coeffs, self.sh_monomials)
    
        sph = jnp.where(edge_mask[..., None], sph, 0.0)
        atom_e = jnp.zeros(species.shape[0], edge_vectors.dtype)
    
        for layer in self.layers:
            h0, h1, e = layer(h0, h1, species, sph, radial_basis, senders, edge_mask)
            atom_e = atom_e + e
        return atom_e + self.offsets[species].astype(atom_e.dtype)

    def __call__(
        self,
        positions,
        species,
        *,
        box_vectors=None,
        neighbors=None,
        neighbor_idx=None,
        periodic: bool | None = False,
        extra_capacity: int = 0,
    ):
        periodic = bool(periodic)
        if neighbor_idx is None:
            neighbors = get_neighbors(
                positions,
                box_vectors if periodic else None,
                cutoff=float(self.cutoff),
                cell_atom_threshold=int(self.neighbor_cell_atom_threshold),
                cell_capacity_multiplier=float(self.neighbor_cell_capacity_multiplier),
                extra_capacity=extra_capacity,
                neighbors=neighbors,
                periodic=periodic,
            )
            neighbor_idx = neighbors.idx

        return jnp.sum(
            self.node_energies(
                positions,
                species,
                neighbor_idx=neighbor_idx,
                box_vectors=box_vectors if periodic else None,
            )
        )


def save_model(
    path: str | PathLike,
    model: MACEModel,
    *,
    name: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    if not model.layers:
        raise ValueError("Cannot save a MACE model with no layers.")
    model_config = dict(config) if config is not None else {
        "name": model.name if name is None else name,
        "num_species": model.num_species,
        "cutoff": model.cutoff,
        "num_layers": len(model.layers),
        "num_features": model.num_features,
        "correlation": 3,
        "num_radial_basis": model.num_radial_basis,
        "avg_num_neighbors": 1.0 / float(model.layers[0].epsilon),
        "hidden_has_vector": model.hidden_has_vector,
        "radial_polynomial_p": model.radial_polynomial_p,
        "silu_normalization": model.silu_normalization,
        "neighbor_cell_atom_threshold": model.neighbor_cell_atom_threshold,
        "neighbor_cell_capacity_multiplier": model.neighbor_cell_capacity_multiplier,
        "sh_dims": list(model.sh_dims),
        "sh_starts": list(model.sh_starts),
        "sh_monomials": [list(row) for row in model.sh_monomials],
    }
    with Path(path).open("wb") as handle:
        handle.write((json.dumps(model_config, sort_keys=True) + "\n").encode())
        eqx.tree_serialise_leaves(handle, model)


def load_model(
    model: str | PathLike = "mace-off-m(24)",
    *,
    model_path: str | PathLike | None = None,
    dtype: Any = jnp.float32,
    neighbor_cell_atom_threshold: int | None = None,
    neighbor_cell_capacity_multiplier: float | None = None,
) -> MACEModel:
    if model_path is not None:
        path = Path(model_path)
    elif isinstance(model, PathLike):
        path = Path(model)
    else:
        if model not in MACE_MODEL_PATHS:
            choices = ", ".join(sorted(MACE_MODEL_PATHS))
            raise ValueError(f"Unknown MACE model {model!r}. Expected one of: {choices}")
        path = MACE_MODEL_PATHS[model]

    with path.open("rb") as handle:
        config = json.loads(handle.readline().decode())
        config = dict(config)
        if neighbor_cell_atom_threshold is not None:
            config["neighbor_cell_atom_threshold"] = int(neighbor_cell_atom_threshold)
        if neighbor_cell_capacity_multiplier is not None:
            config["neighbor_cell_capacity_multiplier"] = float(
                neighbor_cell_capacity_multiplier
            )
        template = MACEModel(
            np.zeros((config["num_species"],), dtype=np.float32),
            config,
            dtype=jnp.float32,
            key=jax.random.PRNGKey(0),
        )
        loaded = eqx.tree_deserialise_leaves(handle, template)
    if dtype == jnp.float32:
        return loaded
    return jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating) else x,
        loaded,
    )
