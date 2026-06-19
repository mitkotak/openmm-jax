# Credit to https://github.com/isayevlab/aimnetcentral for the docs

from __future__ import annotations

import json
import math
from os import PathLike
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import erfc
from jax_md import partition, space

jax.config.update("jax_enable_x64", True)

AIMNET2_MODEL_PATHS = {
    "aimnet2-jax": Path(__file__).resolve().with_name("aimnet2.eqx"),
}
AIMNET2_MODEL_NAMES = tuple(AIMNET2_MODEL_PATHS)


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
        return neighbors.update(positions, **neighbor_kwargs)

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
    return neighbor_fn.allocate(
        positions,
        extra_capacity=int(extra_capacity),
        **neighbor_kwargs,
    )


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
    neighbor_mask = (neighbors >= 0) & (neighbors < num_atoms)
    safe_neighbors = jnp.where(neighbor_mask, neighbors, atom_ids[:, None])

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
    edge_mask = neighbor_mask & (safe_neighbors != atom_ids[:, None]) & (distances > 1.0e-8)
    if cutoff is not None:
        edge_mask = edge_mask & (distances < cutoff)
    edge_vectors = jnp.where(edge_mask[..., None], edge_vectors, 0.0)
    return edge_vectors, safe_neighbors, edge_mask


def safe_norm(x: Array, *, axis=-1, keepdims: bool = False, eps: float = 1.0e-24) -> Array:
    return jnp.sqrt(jnp.maximum(jnp.sum(x * x, axis=axis, keepdims=keepdims), eps))


class MLP(eqx.Module):
    layers: list["Linear"]

    def __init__(self, sizes: tuple[int, ...], *, dtype: Any = jnp.float32, key: Array):
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [
            Linear(in_dim, out_dim, dtype=dtype, key=subkey)
            for subkey, in_dim, out_dim in zip(keys, sizes[:-1], sizes[1:])
        ]

    def __call__(self, x: Array, *, last_linear: bool = True) -> Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or not last_linear:
                x = jax.nn.gelu(x, approximate=False)
        return x


class Linear(eqx.Module):
    weight: Array
    bias: Array
    in_dim: int = eqx.field(static=True)

    def __init__(self, in_dim: int, out_dim: int, *, dtype: Any = jnp.float32, key: Array):
        weight_key, bias_key = jax.random.split(key)
        lim = jnp.sqrt(1.0 / in_dim)
        self.weight = jax.random.uniform(
            weight_key,
            (out_dim, in_dim),
            dtype=dtype,
            minval=-lim,
            maxval=lim,
        )
        self.bias = jax.random.uniform(
            bias_key,
            (out_dim,),
            dtype=dtype,
            minval=-lim,
            maxval=lim,
        )
        self.in_dim = in_dim

    def __call__(self, x: Array) -> Array:
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected feature axis of size {self.in_dim}, got shape {x.shape}."
            )
        return x @ self.weight.T + self.bias


def d3bj_energy_neighbors(
    positions: Array,
    d3_pre: dict[str, Array],
    neighbor_idx: Array,
    box_vectors: Array | None = None,
    *,
    cutoff: float,
    smoothing_fraction: float,
) -> Array:
    edge_vectors, safe_neighbors, edge_mask = dense_neighbor_edges(
        positions,
        neighbor_idx,
        box_vectors=box_vectors,
        cutoff=float(cutoff),
    )
    distances = safe_norm(edge_vectors, axis=-1)
    rij = distances / float(d3_pre["bohr_a"])
    sp_idx = d3_pre["species_idx"]
    sp_i = sp_idx[:, None]
    sp_j = sp_idx[safe_neighbors]
    rcov = d3_pre["rcov"]
    r2r4 = d3_pre["r2r4"]

    rr = (rcov[sp_i] + rcov[sp_j]) / jnp.maximum(rij, 1.0e-8)
    damp = 1.0 / (1.0 + jnp.exp(-float(d3_pre["d3_k1"]) * (rr - 1.0)))
    cn = jnp.sum(jnp.where(edge_mask, damp, 0.0), axis=1)

    atom_ids = jnp.arange(positions.shape[0], dtype=jnp.int32)
    pair_mask = edge_mask & (atom_ids[:, None] < safe_neighbors)
    pair_c6ab = d3_pre["c6ab"][sp_i, sp_j]
    e_pair = _d3_pair_energy(
        pair_c6ab,
        cn[:, None],
        cn[safe_neighbors],
        rij,
        r2r4[sp_i],
        r2r4[sp_j],
        d3_s6=float(d3_pre["d3_s6"]),
        d3_s8=float(d3_pre["d3_s8"]),
        d3_a1=float(d3_pre["d3_a1"]),
        d3_a2=float(d3_pre["d3_a2"]),
        d3_k3=float(d3_pre["d3_k3"]),
    )
    switch = _s5_switch(
        rij,
        smoothing_on=float(cutoff) * (1.0 - float(smoothing_fraction)) / float(d3_pre["bohr_a"]),
        smoothing_off=float(cutoff) / float(d3_pre["bohr_a"]),
    )
    return (
        jnp.sum(jnp.where(pair_mask, (e_pair * switch).astype(jnp.float64), 0.0))
        * float(d3_pre["hartree_ev"])
    )


def _s5_switch(distance: Array, *, smoothing_on: float, smoothing_off: float) -> Array:
    if smoothing_off <= smoothing_on:
        return jnp.ones_like(distance)
    t = jnp.clip((distance - smoothing_on) / (smoothing_off - smoothing_on), 0.0, 1.0)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    switch = 1.0 - (10.0 * t3 - 15.0 * t4 + 6.0 * t5)
    return jnp.where(distance <= smoothing_on, 1.0, switch)


def _d3_pair_energy(
    pair_c6ab: Array,
    nci: Array,
    ncj: Array,
    rij: Array,
    r2r4_i: Array,
    r2r4_j: Array,
    *,
    d3_s6: float,
    d3_s8: float,
    d3_a1: float,
    d3_a2: float,
    d3_k3: float,
) -> Array:
    reference_c6 = pair_c6ab[..., 0]
    reference_cn_i = pair_c6ab[..., 1]
    reference_cn_j = pair_c6ab[..., 2]
    num_cn_references = reference_c6.shape[-2] * reference_c6.shape[-1]

    cn_distance2 = (reference_cn_i - nci[..., None, None]) ** 2 + (
        reference_cn_j - ncj[..., None, None]
    ) ** 2
    reference_logits = jnp.where(
        reference_c6 > 0.0,
        d3_k3 * cn_distance2,
        -1.0e20,
    ).reshape(*rij.shape, num_cn_references)

    reference_weights = jax.nn.softmax(reference_logits, axis=-1)
    reference_c6 = reference_c6.reshape(*rij.shape, num_cn_references)
    c6 = jnp.sum(reference_weights * reference_c6, axis=-1)
    c8 = 3.0 * c6 * r2r4_i * r2r4_j

    bj_radius = (
        d3_a1 * jnp.sqrt(jnp.maximum(c8 / jnp.maximum(c6, 1.0e-30), 0.0))
        + d3_a2
    )
    bj_radius2 = bj_radius**2
    bj_radius6 = bj_radius2**3
    bj_radius8 = bj_radius6 * bj_radius2

    e6 = -d3_s6 * c6 / (rij**6 + bj_radius6)
    e8 = -d3_s8 * c8 / (rij**8 + bj_radius8)
    return e6 + e8


def radial_symmetry_functions(distance: Array, shifts: Array, eta: Array, cutoff: float) -> Array:
    cutoff_values = cosine_cutoff(distance, cutoff)
    return jnp.exp(-eta * (distance[..., None] - shifts) ** 2) * cutoff_values[..., None]


def cosine_cutoff(distance: Array, cutoff: float) -> Array:
    distance = jnp.clip(distance, 1.0e-6, cutoff)
    return 0.5 * (jnp.cos(distance * jnp.pi / cutoff) + 1.0)


def _exp_cutoff(d: Array, rc: float, exp_minus_1: float) -> Array:
    x = jnp.clip(d / rc, 0.0, 1.0 - 1.0e-6)
    return jnp.exp(-1.0 / (1.0 - x**2)) / exp_minus_1


def atomic_embedding_features(
    atomic_embeddings: Array, g_ijs: Array, unit_vectors: Array, vector_weights: Array,
    neighbors: Array, edge_mask: Array,
) -> Array:
    neighbor_embeddings = atomic_embeddings[neighbors]
    neighbor_embeddings = jnp.where(edge_mask[..., None, None], neighbor_embeddings, 0.0)
    scalar_features = jnp.sum(neighbor_embeddings * g_ijs[:, :, None, :], axis=1)
    vector_features = jnp.einsum(
        "nkag,nkg,nkd->nagd",
        neighbor_embeddings,
        g_ijs,
        unit_vectors,
    )
    num_atoms = atomic_embeddings.shape[0]
    scalar_features = scalar_features.reshape(num_atoms, -1)
    vector_features = jnp.einsum("agh,nagd->nahd", vector_weights, vector_features)
    vector_features = jnp.sum(vector_features**2, axis=-1).reshape(num_atoms, -1)
    return jnp.concatenate([scalar_features, vector_features], axis=-1)


def charge_features(
    partial_charges: Array, g_ijs: Array, unit_vectors: Array, vector_weights: Array,
    neighbors: Array, edge_mask: Array,
) -> Array:
    neighbor_charges = partial_charges[neighbors]
    neighbor_charges = jnp.where(edge_mask[..., None], neighbor_charges, 0.0)
    scalar_features = jnp.einsum("nka,nkg->nag", neighbor_charges, g_ijs)
    vector_features = jnp.einsum("nka,nkg,nkd->nagd", neighbor_charges, g_ijs, unit_vectors)
    num_atoms = partial_charges.shape[0]
    scalar_features = scalar_features.reshape(num_atoms, -1)
    vector_features = jnp.einsum("agh,nagd->nahd", vector_weights, vector_features)
    vector_features = jnp.sum(vector_features**2, axis=-1).reshape(num_atoms, -1)
    return jnp.concatenate([scalar_features, vector_features], axis=-1)


def neural_charge_equilibration(
    partial_charges: Array, charge_weights: Array, total_charge: Array | float = 0.0,
) -> Array:
    weights = charge_weights**2
    weight_sum = jnp.sum(weights, axis=0, keepdims=True) + 1.0e-6
    predicted_charge = jnp.sum(partial_charges, axis=0, keepdims=True)
    return partial_charges + (weights / weight_sum) * (total_charge - predicted_charge)


def _short_range_coulomb_dense(
    charges: Array,
    d: Array,
    neighbors: Array,
    edge_mask: Array,
    *,
    coulomb_rc: float,
    coulomb_factor: float,
    exp_minus_1: float,
) -> Array:
    q_ij = charges[:, None] * charges[neighbors]
    inv_d = 1.0 / jnp.maximum(d, 1.0e-8)
    fc = _exp_cutoff(d, coulomb_rc, exp_minus_1)
    e = coulomb_factor * (fc * q_ij * inv_d).astype(jnp.float64)
    return jnp.sum(jnp.where(edge_mask, e, 0.0))


def _dsf_coulomb_dense(
    charges: Array,
    positions: Array,
    neighbor_idx: Array,
    *,
    box_vectors: Array,
    cutoff: float,
    alpha: float,
    coulomb_factor: float,
) -> Array:
    edge_vectors, safe_neighbors, edge_mask = dense_neighbor_edges(
        positions,
        neighbor_idx,
        box_vectors=box_vectors,
        cutoff=float(cutoff),
    )
    d = safe_norm(edge_vectors, axis=-1)
    rc = float(cutoff)
    erfc_alpha_rc = float(math.erfc(float(alpha) * rc))
    c2 = erfc_alpha_rc / rc
    c3 = c2 / rc
    c4 = 2.0 * float(alpha) * math.exp(-((float(alpha) * rc) ** 2)) / (
        rc * math.sqrt(math.pi)
    )
    j_dsf = erfc(float(alpha) * d) / jnp.maximum(d, 1.0e-8)
    j_dsf = j_dsf - c2 + (d - rc) * (c3 + c4)
    q_ij = charges[:, None] * charges[safe_neighbors]
    e = coulomb_factor * (q_ij * j_dsf).astype(jnp.float64)
    return jnp.sum(jnp.where(edge_mask, e, 0.0))


def _simple_coulomb_all_pairs(
    positions: Array,
    charges: Array,
    hartree_bohr: float,
) -> Array:
    num_atoms = positions.shape[0]
    delta = positions[:, None, :] - positions[None, :, :]
    distance = jnp.sqrt(jnp.maximum(jnp.sum(delta * delta, axis=-1), 1.0e-12))
    pair_mask = jnp.arange(num_atoms)[:, None] < jnp.arange(num_atoms)[None, :]
    pair_energy = hartree_bohr * charges[:, None] * charges[None, :] / jnp.maximum(
        distance,
        1.0e-8,
    )
    return jnp.sum(jnp.where(pair_mask, pair_energy, 0.0).astype(jnp.float64))


def aimnet2_message_passing(
    model,
    species,
    unit_vectors,
    g_ijs,
    neighbors,
    edge_mask,
    total_charge,
):
    num_atoms = species.shape[0]
    nfeature, nshifts, ncharge = model.nfeature, model.nshifts, model.ncharge

    def embedding_features(atomic_embeddings):
        return atomic_embedding_features(
            atomic_embeddings, g_ijs, unit_vectors, model.conv_a_agh, neighbors, edge_mask
        )

    def partial_charge_features(partial_charges):
        return charge_features(
            partial_charges, g_ijs, unit_vectors, model.conv_q_agh, neighbors, edge_mask
        )

    atomic_embeddings = model.afv[species].reshape(num_atoms, nfeature, nshifts)
    embedding_flat = atomic_embeddings.reshape(num_atoms, -1)
    out0 = model.mlp0(
        jnp.concatenate([embedding_flat, embedding_features(atomic_embeddings)], axis=-1),
        last_linear=model.mlp_last_linear[0],
    )
    partial_charges = neural_charge_equilibration(
        out0[:, :ncharge], out0[:, ncharge : 2 * ncharge], total_charge
    )
    atomic_embeddings = (embedding_flat + out0[:, 2 * ncharge :]).reshape(
        num_atoms, nfeature, nshifts
    )

    embedding_flat = atomic_embeddings.reshape(num_atoms, -1)
    out1 = model.mlp1(
        jnp.concatenate([
            embedding_flat,
            embedding_features(atomic_embeddings),
            partial_charges,
            partial_charge_features(partial_charges),
        ], axis=-1),
        last_linear=model.mlp_last_linear[1],
    )
    partial_charges = neural_charge_equilibration(
        partial_charges + out1[:, :ncharge], out1[:, ncharge : 2 * ncharge], total_charge
    )
    atomic_embeddings = (embedding_flat + out1[:, 2 * ncharge :]).reshape(
        num_atoms, nfeature, nshifts
    )

    embedding_flat = atomic_embeddings.reshape(num_atoms, -1)
    aim_vectors = model.mlp2(
        jnp.concatenate([
            embedding_flat,
            embedding_features(atomic_embeddings),
            partial_charges,
            partial_charge_features(partial_charges),
        ], axis=-1),
        last_linear=model.mlp_last_linear[2],
    )
    return aim_vectors, partial_charges


def aimnet2_coulomb_energy(
    model,
    partial_charges,
    positions,
    r_ij,
    neighbors,
    edge_mask,
    lr_neighbor_idx,
    box_vectors,
):
    partial_charges = partial_charges.squeeze(-1)
    local_coulomb = _short_range_coulomb_dense(
        partial_charges,
        r_ij,
        neighbors,
        edge_mask,
        coulomb_rc=model.coulomb_rc,
        coulomb_factor=model.coulomb_factor,
        exp_minus_1=model.exp_minus_1,
    )
    if box_vectors is None:
        total_coulomb = _simple_coulomb_all_pairs(
            positions,
            partial_charges,
            hartree_bohr=model.hartree_bohr,
        )
    else:
        total_coulomb = _dsf_coulomb_dense(
            partial_charges,
            positions,
            lr_neighbor_idx,
            box_vectors=box_vectors,
            cutoff=float(model.lr_cutoff),
            alpha=float(model.dsf_alpha),
            coulomb_factor=model.coulomb_factor,
        )
    return total_coulomb - local_coulomb


class AIMNet2Model(eqx.Module):
    bohr_a: float = eqx.field(static=True)
    coulomb_factor: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    nfeature: int = eqx.field(static=True)
    nshifts: int = eqx.field(static=True)
    ncharge: int = eqx.field(static=True)
    ncomb_v: int = eqx.field(static=True)
    coulomb_rc: float = eqx.field(static=True)
    d3_k1: float = eqx.field(static=True)
    d3_k3: float = eqx.field(static=True)
    ev_to_kjmol: float = eqx.field(static=True)
    exp_minus_1: float = eqx.field(static=True)
    hartree_bohr: float = eqx.field(static=True)
    hartree_ev: float = eqx.field(static=True)
    mlp_last_linear: tuple[bool, ...] = eqx.field(static=True)
    implemented_species: tuple[int, ...] = eqx.field(static=True)
    d3_s6: float = eqx.field(static=True)
    d3_s8: float = eqx.field(static=True)
    d3_a1: float = eqx.field(static=True)
    d3_a2: float = eqx.field(static=True)
    neighbor_cell_atom_threshold: int = eqx.field(static=True)
    neighbor_cell_capacity_multiplier: float = eqx.field(static=True)
    lr_cutoff: float = eqx.field(static=True)
    d3_smoothing_fraction: float = eqx.field(static=True)
    dsf_alpha: float = eqx.field(static=True)
    afv: Array
    shifts: Array
    eta: Array
    conv_a_agh: Array
    conv_q_agh: Array
    mlp0: MLP
    mlp1: MLP
    mlp2: MLP
    energy_mlp: MLP
    atomic_shifts: Array
    d3_c6ab: Array | None
    d3_rcov: Array | None
    d3_r2r4: Array | None

    def __init__(
        self,
        *,
        config: dict[str, Any],
        d3_params: dict[str, np.ndarray | Array] | None = None,
        dtype: Any = jnp.float32,
        key: Array = jax.random.PRNGKey(0),
    ):
        keys = jax.random.split(key, 4)
        self.bohr_a = float(config["bohr_a"])
        self.coulomb_factor = float(config["coulomb_factor"])
        self.cutoff = float(config["cutoff"])
        self.nfeature = int(config["nfeature"])
        self.nshifts = int(config["nshifts"])
        self.ncharge = int(config["ncharge"])
        self.ncomb_v = int(config["ncomb_v"])
        self.coulomb_rc = float(config["coulomb_rc"])
        self.d3_k1 = float(config["d3_k1"])
        self.d3_k3 = float(config["d3_k3"])
        self.ev_to_kjmol = float(config["ev_to_kjmol"])
        self.exp_minus_1 = float(config["exp_minus_1"])
        self.hartree_bohr = float(config["hartree_bohr"])
        self.hartree_ev = float(config["hartree_ev"])
        self.mlp_last_linear = tuple(bool(x) for x in config["mlp_last_linear"])
        self.implemented_species = tuple(int(x) for x in config["implemented_species"])
        self.d3_s6 = float(config["d3_s6"])
        self.d3_s8 = float(config["d3_s8"])
        self.d3_a1 = float(config["d3_a1"])
        self.d3_a2 = float(config["d3_a2"])
        self.neighbor_cell_atom_threshold = int(config["neighbor_cell_atom_threshold"])
        self.neighbor_cell_capacity_multiplier = float(
            config["neighbor_cell_capacity_multiplier"]
        )
        self.lr_cutoff = float(config["lr_cutoff"])
        self.d3_smoothing_fraction = float(config["d3_smoothing_fraction"])
        self.dsf_alpha = float(config["dsf_alpha"])
        self.afv = jnp.zeros((64, self.nfeature * self.nshifts), dtype=dtype)
        self.shifts = jnp.zeros((self.nshifts,), dtype=dtype)
        self.eta = jnp.zeros((), dtype=dtype)
        self.conv_a_agh = jnp.zeros((self.nfeature, self.nshifts, self.ncomb_v), dtype=dtype)
        self.conv_q_agh = jnp.zeros((self.ncharge, self.nshifts, self.ncomb_v), dtype=dtype)
        self.mlp0 = MLP(config["mlp0_sizes"], dtype=dtype, key=keys[0])
        self.mlp1 = MLP(config["mlp1_sizes"], dtype=dtype, key=keys[1])
        self.mlp2 = MLP(config["mlp2_sizes"], dtype=dtype, key=keys[2])
        self.energy_mlp = MLP(config["energy_sizes"], dtype=dtype, key=keys[3])
        self.atomic_shifts = jnp.zeros((64,), dtype=jnp.float64)
        if d3_params is None and config["has_d3_params"]:
            try:
                c6ab_shape = tuple(int(v) for v in config["d3_c6ab_shape"])
                rcov_shape = tuple(int(v) for v in config["d3_rcov_shape"])
                r2r4_shape = tuple(int(v) for v in config["d3_r2r4_shape"])
            except KeyError as exc:
                raise ValueError(
                    "AIMNet2 checkpoint config says it has D3 params but is missing D3 shapes."
                ) from exc
            self.d3_c6ab = jnp.zeros(c6ab_shape, dtype=dtype)
            self.d3_rcov = jnp.zeros(rcov_shape, dtype=dtype)
            self.d3_r2r4 = jnp.zeros(r2r4_shape, dtype=dtype)
        elif d3_params is None:
            self.d3_c6ab = None
            self.d3_rcov = None
            self.d3_r2r4 = None
        else:
            self.d3_c6ab = jnp.asarray(d3_params["c6ab"], dtype=dtype)
            self.d3_rcov = jnp.asarray(d3_params["rcov"], dtype=dtype)
            self.d3_r2r4 = jnp.asarray(d3_params["r2r4"], dtype=dtype)

    def node_energies_and_charges(
        self,
        positions: Array,
        species: Array,
        *,
        neighbor_idx: Array,
        total_charge: Array | float = 0.0,
        box_vectors: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Return local node energies plus intermediates needed by global terms."""

        species = jnp.asarray(species, dtype=jnp.int32)
        local_vectors, neighbors, edge_mask = dense_neighbor_edges(
            positions,
            neighbor_idx,
            box_vectors=box_vectors,
            cutoff=float(self.cutoff),
        )
        r_ij = safe_norm(local_vectors, axis=-1)
        unit_vectors = local_vectors / jnp.maximum(r_ij[..., None], 1.0e-8)
        g_ijs = radial_symmetry_functions(r_ij, self.shifts, self.eta, self.cutoff)
        g_ijs = jnp.where(edge_mask[..., None], g_ijs, 0.0)
        aim_vectors, partial_charges = aimnet2_message_passing(
            self,
            species,
            unit_vectors,
            g_ijs,
            neighbors,
            edge_mask,
            total_charge,
        )

        atom_local_energy = self.energy_mlp(aim_vectors, last_linear=True).squeeze(-1)
        node_energies = atom_local_energy.astype(jnp.float64) + self.atomic_shifts[species]
        return node_energies, partial_charges, r_ij, neighbors, edge_mask

    def __call__(
        self,
        positions: Array,
        species: Array,
        *,
        d3_data: dict[str, Array],
        box_vectors: Array | None = None,
        neighbors=None,
        neighbor_idx: Array | None = None,
        lr_neighbors=None,
        lr_neighbor_idx: Array | None = None,
        periodic: bool | None = False,
        extra_capacity: int = 0,
        total_charge: Array | float = 0.0,
    ) -> Array:
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
        if lr_neighbor_idx is None:
            lr_neighbors = get_neighbors(
                positions,
                box_vectors if periodic else None,
                cutoff=float(self.lr_cutoff),
                cell_atom_threshold=int(self.neighbor_cell_atom_threshold),
                cell_capacity_multiplier=float(self.neighbor_cell_capacity_multiplier),
                extra_capacity=extra_capacity,
                neighbors=lr_neighbors,
                periodic=periodic,
            )
            lr_neighbor_idx = lr_neighbors.idx

        box_vectors = box_vectors if periodic else None
        node_energies, partial_charges, r_ij, neighbors, edge_mask = self.node_energies_and_charges(
                                                                    positions,
                                                                    species,
                                                                    neighbor_idx=neighbor_idx,
                                                                    total_charge=total_charge,
                                                                    box_vectors=box_vectors,
                                                                )
        local_energy = jnp.sum(node_energies)
        coulomb_energy = aimnet2_coulomb_energy(
            self,
            partial_charges,
            positions,
            r_ij,
            neighbors,
            edge_mask,
            lr_neighbor_idx,
            box_vectors,
        )
        dispersion_energy = d3bj_energy_neighbors(
            positions,
            d3_data,
            lr_neighbor_idx,
            box_vectors=box_vectors,
            cutoff=float(self.lr_cutoff),
            smoothing_fraction=float(self.d3_smoothing_fraction),
        ).astype(jnp.float64)

        total_energy = local_energy + coulomb_energy + dispersion_energy
        return total_energy.astype(jnp.float32)


def load_aimnet2_model(
    model: str | PathLike = "aimnet2-jax",
    *,
    model_path: str | PathLike | None = None,
) -> AIMNet2Model:
    if model_path is not None:
        path = Path(model_path)
    elif isinstance(model, PathLike):
        path = Path(model)
    elif model in AIMNET2_MODEL_PATHS:
        path = AIMNET2_MODEL_PATHS[model]
    else:
        path = Path(model)
    if not path.is_file():
        raise FileNotFoundError(f"AIMNet2 .eqx checkpoint not found: {path}")
    with path.open("rb") as handle:
        config = json.loads(handle.readline().decode())
        template = AIMNet2Model(config=config, dtype=jnp.float32)
        return eqx.tree_deserialise_leaves(handle, template)
