# Credit to https://github.com/torchmd/torchmd-net

import json
import pickle
from os import PathLike
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition, space

jax.config.update("jax_default_matmul_precision", "highest")

ACEFF_MODEL_PATHS = {
    "aceff-jax-1.1": Path(__file__).resolve().with_name("aceff_v1.1.eqx"),
    "aceff-jax-2.0": Path(__file__).resolve().with_name("aceff_v2.0.eqx"),
}
ACEFF_MODEL_NAMES = tuple(ACEFF_MODEL_PATHS)


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
    use_cell_list = periodic and num_atoms >= int(cell_atom_threshold)
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


def dense_neighbor_edges(positions, neighbors, *, cutoff: float, box_vectors=None):
    num_atoms = int(positions.shape[0])
    atom_ids = jnp.arange(num_atoms, dtype=jnp.int32)
    neighbor_idx = jnp.asarray(
        neighbors.idx if hasattr(neighbors, "idx") else neighbors,
        dtype=jnp.int32,
    )
    valid = (neighbor_idx >= 0) & (neighbor_idx < num_atoms)
    valid = valid & (neighbor_idx != atom_ids[:, None])
    fallback_dst = jnp.where(num_atoms > 1, (atom_ids + 1) % num_atoms, atom_ids)
    safe_dst = jnp.where(valid, neighbor_idx, fallback_dst[:, None])

    edge_src = jnp.concatenate([atom_ids, jnp.repeat(atom_ids, neighbor_idx.shape[1])])
    edge_dst = jnp.concatenate([atom_ids, safe_dst.reshape((-1,))])
    edge_mask = jnp.concatenate([jnp.ones((num_atoms,), dtype=bool), valid.reshape((-1,))])

    raw_vec = positions[edge_src] - positions[edge_dst]
    if box_vectors is None:
        pbc_shifts = jnp.zeros_like(raw_vec)
    else:
        displacement, _ = space.periodic_general(
            jnp.swapaxes(jnp.asarray(box_vectors, dtype=positions.dtype), -1, -2),
            fractional_coordinates=True,
        )
        edge_vec = jax.vmap(displacement)(positions[edge_src], positions[edge_dst])
        pbc_shifts = edge_vec - raw_vec

    far_shift = jnp.zeros_like(pbc_shifts).at[:, 0].set(float(cutoff) + 1.0)
    pbc_shifts = jnp.where(edge_mask[:, None], pbc_shifts, far_shift - raw_vec)
    return edge_src, edge_dst, pbc_shifts, edge_mask


def unique_pairs(num_atoms: int):
    pair_src, pair_dst = np.triu_indices(int(num_atoms), k=1)
    return (
        jnp.asarray(pair_src, dtype=jnp.int32),
        jnp.asarray(pair_dst, dtype=jnp.int32),
    )


def cosine_cutoff(d, cutoff, cutoff_lower=0.0):
    if cutoff_lower > 0:
        x = 2.0 * (d - cutoff_lower) / (cutoff - cutoff_lower) + 1.0
        c = 0.5 * (jnp.cos(jnp.pi * x) + 1.0)
        return jnp.where((d < cutoff) & (d > cutoff_lower), c, 0.0)
    else:
        c = 0.5 * (jnp.cos(d * jnp.pi / cutoff) + 1.0)
        return jnp.where(d < cutoff, c, 0.0)


def _isotropic_to_tensor(isotropic):
    """Scalar [N, F] -> diagonal tensor [N, 3, 3, F]."""
    eye = jnp.eye(3, dtype=isotropic.dtype)[None, :, :, None]
    return isotropic[:, None, None, :] * eye


def decompose_tensor(X):
    """Decompose into irreducible components I, A, S."""
    antisymmetric = 0.5 * (X - jnp.swapaxes(X, 1, 2))
    symmetric_full = X - antisymmetric
    isotropic = jnp.diagonal(X, axis1=1, axis2=2).mean(axis=-1)
    symmetric = symmetric_full - _isotropic_to_tensor(isotropic)
    return isotropic, antisymmetric, symmetric


def compose_tensor(isotropic, antisymmetric, symmetric):
    return _isotropic_to_tensor(isotropic) + antisymmetric + symmetric


def tensor_norm(X):
    """Frobenius norm squared: [N, 3, 3, F] -> [N, F]."""
    return (X**2).sum(axis=(1, 2))


def vector_to_skewtensor(vec):
    """[N, 3, F] -> [N, 3, 3, F] skew-symmetric."""
    N, _, F = vec.shape
    zero = jnp.zeros((N, F), dtype=vec.dtype)
    tensor = jnp.stack(
        [
            zero,
            -vec[:, 2, :],
            vec[:, 1, :],
            vec[:, 2, :],
            zero,
            -vec[:, 0, :],
            -vec[:, 1, :],
            vec[:, 0, :],
            zero,
        ],
        axis=1,
    )
    return tensor.reshape(N, 3, 3, F)


def skewtensor_to_vector(A):
    """[N, 3, 3, F] -> [N, 3, F]."""
    A_flat = A.reshape(A.shape[0], 9, A.shape[3])
    return 0.5 * jnp.stack(
        [
            A_flat[:, 7] - A_flat[:, 5],
            A_flat[:, 2] - A_flat[:, 6],
            A_flat[:, 3] - A_flat[:, 1],
        ],
        axis=1,
    )


def outer_to_symtensor(T):
    """Symmetrize and remove trace. [N, 3, 3, F] -> [N, 3, 3, F]."""
    symmetric = 0.5 * (T + jnp.swapaxes(T, 1, 2))
    isotropic = jnp.diagonal(T, axis1=1, axis2=2).mean(axis=-1)
    return symmetric - _isotropic_to_tensor(isotropic)


def tensor_matmul_o3(Y, msg):
    """O(3)-equivariant contraction: Y*msg + msg*Y."""
    Yp = jnp.transpose(Y, (0, 3, 1, 2))
    Mp = jnp.transpose(msg, (0, 3, 1, 2))
    return jnp.transpose(Mp @ Yp, (0, 2, 3, 1)) + jnp.transpose(Yp @ Mp, (0, 2, 3, 1))


def tensor_matmul_so3(Y, msg):
    """SO(3)-equivariant contraction: Y*msg."""
    Yp = jnp.transpose(Y, (0, 3, 1, 2))
    Mp = jnp.transpose(msg, (0, 3, 1, 2))
    return jnp.transpose(Yp @ Mp, (0, 2, 3, 1))


class Linear(eqx.Module):
    kernel: Any
    bias: Any

    def __init__(self, kernel, bias=None):
        self.kernel = kernel
        self.bias = bias

    def __call__(self, x):
        out = x @ self.kernel
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(eqx.Module):
    weight: Any
    bias: Any
    eps: float = eqx.field(static=True)

    def __init__(self, weight, bias, *, eps: float = 1.0e-5):
        self.weight = weight
        self.bias = bias
        self.eps = float(eps)

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return self.weight * (x - mean) * jax.lax.rsqrt(var + self.eps) + self.bias


def safe_norm(x, *, axis=-1, keepdims: bool = False, eps: float = 1.0e-24):
    return jnp.sqrt(jnp.maximum(jnp.sum(x * x, axis=axis, keepdims=keepdims), eps))


def embedding(
    species,
    edge_src,
    edge_dst,
    distances,
    unit_vectors,
    radial_features,
    cutoff,
    cutoff_lower,
    weights,
    edge_mask=None,
):
    num_features = weights["emb"].shape[1]
    num_atoms = species.shape[0]

    Zi = weights["emb"][species[edge_src]]
    Zj = weights["emb"][species[edge_dst]]
    Zij = weights["emb2"](
        jnp.concatenate([Zi, Zj], axis=-1),
    )

    distance_projection_1 = weights["distance_proj1"](radial_features)
    distance_projection_2 = weights["distance_proj2"](radial_features)
    distance_projection_3 = weights["distance_proj3"](radial_features)

    cutoff_values = cosine_cutoff(distances, cutoff, cutoff_lower)
    if edge_mask is not None:
        cutoff_values = cutoff_values * edge_mask
    species_pair_features = cutoff_values[:, None] * Zij

    edge_features = species_pair_features[:, None, :] * jnp.stack(
        [distance_projection_1, distance_projection_2, distance_projection_3],
        axis=1,
    )

    isotropic_messages = edge_features[:, 0, :]
    isotropic = jnp.zeros((num_atoms, num_features), dtype=isotropic_messages.dtype)
    isotropic = isotropic.at[edge_src].add(isotropic_messages)

    antisymmetric_messages = edge_features[:, 1, :][:, None, :] * unit_vectors[:, :, None]
    antisymmetric_vectors = jnp.zeros(
        (num_atoms, 3, num_features),
        dtype=isotropic_messages.dtype,
    )
    antisymmetric_vectors = antisymmetric_vectors.at[edge_src].add(antisymmetric_messages)

    outer = unit_vectors[:, :, None] * unit_vectors[:, None, :]
    symmetric_messages = edge_features[:, 2, :][:, None, None, :] * outer[:, :, :, None]
    symmetric = jnp.zeros(
        (num_atoms, 3, 3, num_features),
        dtype=isotropic_messages.dtype,
    )
    symmetric = symmetric.at[edge_src].add(symmetric_messages)

    antisymmetric = vector_to_skewtensor(antisymmetric_vectors)
    symmetric = outer_to_symtensor(symmetric)
    tensor_features = compose_tensor(isotropic, antisymmetric, symmetric)

    norm = tensor_norm(tensor_features)
    norm = weights["init_norm"](norm)
    for layer in weights["linears_scalar"]:
        norm = jax.nn.silu(layer(norm))
    norm = norm.reshape(-1, 3, num_features)

    isotropic_norm = norm[:, 0, :]
    antisymmetric_norm = norm[:, 1, :][:, None, None, :]
    symmetric_norm = norm[:, 2, :][:, None, None, :]

    isotropic = weights["linears_tensor"][0](isotropic) * isotropic_norm
    antisymmetric = weights["linears_tensor"][1](antisymmetric) * antisymmetric_norm
    symmetric = weights["linears_tensor"][2](symmetric) * symmetric_norm

    return compose_tensor(isotropic, antisymmetric, symmetric)


def neural_charge_equilibration(partial_charges, charge_weights, total_charge=0.0):
    weights = charge_weights**2
    weight_sum = jnp.sum(weights, axis=0, keepdims=True) + 1.0e-6
    predicted_charge = jnp.sum(partial_charges, axis=0, keepdims=True)
    return partial_charges + (weights / weight_sum) * (total_charge - predicted_charge)


def charge_prediction(tensor_features, weights, total_charge=0.0):
    isotropic, antisymmetric, symmetric = decompose_tensor(tensor_features)
    charge_features = jnp.concatenate(
        [isotropic, tensor_norm(antisymmetric), tensor_norm(symmetric)],
        axis=-1,
    )

    charge_features = weights["q_norm"](charge_features)
    for i, layer in enumerate(weights["q_mlp"]):
        charge_features = layer(charge_features)
        if i < len(weights["q_mlp"]) - 1:
            charge_features = jax.nn.silu(charge_features)

    ncharge = charge_features.shape[-1] // 2
    partial_charges = charge_features[:, :ncharge]
    charge_weights = charge_features[:, ncharge:]
    return neural_charge_equilibration(partial_charges, charge_weights, total_charge)


def message_passing(
    tensor_features,
    partial_charges,
    total_charge,
    edge_src,
    edge_dst,
    distances,
    radial_features,
    weights,
    cutoff,
    cutoff_lower,
    group,
    *,
    edge_charge_features: bool,
    total_charge_interaction_scale: bool,
    edge_mask=None,
):
    num_atoms = tensor_features.shape[0]
    num_features = tensor_features.shape[3]

    cutoff_values = cosine_cutoff(distances, cutoff, cutoff_lower)
    if edge_mask is not None:
        cutoff_values = cutoff_values * edge_mask

    if edge_charge_features:
        source_charges = partial_charges[edge_src]
        neighbor_charges = partial_charges[edge_dst]
        edge_features = jnp.concatenate(
            [radial_features, source_charges, neighbor_charges],
            axis=-1,
        )
    else:
        edge_features = radial_features

    for layer in weights["linears_scalar"]:
        edge_features = jax.nn.silu(layer(edge_features))
    edge_features = (edge_features * cutoff_values[:, None]).reshape(-1, 3, num_features)

    tensor_features = tensor_features / (tensor_norm(tensor_features) + 1)[:, None, None, :]

    isotropic, antisymmetric, symmetric = decompose_tensor(tensor_features)
    isotropic = weights["linears_tensor"][0](isotropic)
    antisymmetric = weights["linears_tensor"][1](antisymmetric)
    symmetric = weights["linears_tensor"][2](symmetric)
    projected_features = compose_tensor(isotropic, antisymmetric, symmetric)

    antisymmetric_vectors = skewtensor_to_vector(antisymmetric)

    scalar_weights = edge_features[:, 0, :]
    vector_weights = edge_features[:, 1, :][:, None, :]
    tensor_weights = edge_features[:, 2, :][:, None, None, :]

    isotropic_messages = scalar_weights * isotropic[edge_dst]
    isotropic_message = jnp.zeros((num_atoms, num_features), dtype=tensor_features.dtype)
    isotropic_message = isotropic_message.at[edge_src].add(isotropic_messages)

    antisymmetric_messages = vector_weights * antisymmetric_vectors[edge_dst]
    antisymmetric_message_vectors = jnp.zeros(
        (num_atoms, 3, num_features),
        dtype=tensor_features.dtype,
    )
    antisymmetric_message_vectors = antisymmetric_message_vectors.at[edge_src].add(
        antisymmetric_messages,
    )

    symmetric_messages = tensor_weights * symmetric[edge_dst]
    symmetric_message = jnp.zeros(
        (num_atoms, 3, 3, num_features),
        dtype=tensor_features.dtype,
    )
    symmetric_message = symmetric_message.at[edge_src].add(symmetric_messages)

    antisymmetric_message = vector_to_skewtensor(antisymmetric_message_vectors)
    messages = compose_tensor(isotropic_message, antisymmetric_message, symmetric_message)

    charge_factor = 1.0
    if total_charge_interaction_scale:
        charge_factor = 1.0 + 0.1 * jnp.asarray(total_charge, dtype=tensor_features.dtype)

    if group == "O(3)":
        updates = charge_factor * tensor_matmul_o3(projected_features, messages)
    else:
        updates = 2 * tensor_matmul_so3(projected_features, messages)

    isotropic_update, antisymmetric_update, symmetric_update = decompose_tensor(updates)

    update_norm = tensor_norm(updates) + 1
    isotropic_update = isotropic_update / update_norm
    antisymmetric_update = antisymmetric_update / update_norm[:, None, None, :]
    symmetric_update = symmetric_update / update_norm[:, None, None, :]

    isotropic_update = weights["linears_tensor"][3](isotropic_update)
    antisymmetric_update = weights["linears_tensor"][4](antisymmetric_update)
    symmetric_update = weights["linears_tensor"][5](symmetric_update)
    delta_features = compose_tensor(isotropic_update, antisymmetric_update, symmetric_update)

    return (
        tensor_features
        + delta_features
        + charge_factor * tensor_matmul_so3(delta_features, delta_features)
    )


def edge_features(
    positions,
    edge_src,
    edge_dst,
    pbc_shifts,
    edge_mask,
    weights,
    cutoff,
    cutoff_lower,
    alpha,
):
    edge_vectors = positions[edge_src] - positions[edge_dst] + pbc_shifts
    is_self = edge_src == edge_dst
    distances = jnp.where(is_self, 0.0, safe_norm(edge_vectors, axis=-1, eps=1.0e-30))
    safe_denom = jnp.where(is_self[:, None], 1.0, jnp.maximum(distances[:, None], 1e-8))
    unit_vectors = edge_vectors / safe_denom
    edge_mask = jnp.asarray(edge_mask, dtype=positions.dtype)
    distances_expanded = distances[..., None]
    cutoff_values = cosine_cutoff(distances_expanded, cutoff, 0.0)
    radial_features = (
        cutoff_values
        * jnp.exp(
            -weights["rbf_betas"]
            * (
                jnp.exp(alpha * (-distances_expanded + cutoff_lower))
                - weights["rbf_means"]
            )
            ** 2
        )
        * edge_mask[:, None]
    )
    return distances, unit_vectors, radial_features, edge_mask


def node_energies(tensor_features, weights):
    isotropic, antisymmetric, symmetric = decompose_tensor(tensor_features)
    energy_features = jnp.concatenate(
        [3.0 * isotropic**2, tensor_norm(antisymmetric), tensor_norm(symmetric)],
        axis=-1,
    )
    energy_features = weights["out_norm"](energy_features)
    energy_features = jax.nn.silu(weights["linear"](energy_features))
    for i, layer in enumerate(weights["output_network"]):
        energy_features = layer(energy_features)
        if i < len(weights["output_network"]) - 1:
            energy_features = jax.nn.silu(energy_features)
    return energy_features.squeeze(-1)


def coulomb_energy(
    model,
    positions,
    partial_charges,
    *,
    box_vectors=None,
):
    partial_charges = jnp.concatenate(partial_charges, axis=-1)
    pair_src, pair_dst = unique_pairs(int(positions.shape[0]))
    if box_vectors is None:
        pair_vectors = positions[pair_src] - positions[pair_dst]
    else:
        displacement, _ = space.periodic_general(
            jnp.swapaxes(jnp.asarray(box_vectors, dtype=positions.dtype), -1, -2),
            fractional_coordinates=True,
        )
        pair_vectors = jax.vmap(displacement)(positions[pair_src], positions[pair_dst])

    distances = safe_norm(pair_vectors, axis=-1)
    damping_x = jnp.clip(
        distances / float(model.coulomb_damp_cutoff),
        0.0,
        1.0 - 1e-6,
    )
    damping = jnp.exp(-1.0 / (1.0 - damping_x**2)) / float(model.exp_minus_1)
    cutoff_values = 1.0 - damping
    charge_products = partial_charges[pair_src] * partial_charges[pair_dst]
    charge_weights = model.params["qweights"]
    weighted_charge_products = (
        charge_products * charge_weights[None, :]
    ).sum(axis=-1) / charge_weights.sum()
    if model.coulomb_cutoff is None:
        pair_energies = cutoff_values * weighted_charge_products / distances
    else:
        cutoff = float(model.coulomb_cutoff)
        epsilon = float(model.coulomb_epsilon_solvent)
        k_rf = (1.0 / cutoff**3) * (epsilon - 1.0) / (2.0 * epsilon + 1.0)
        c_rf = (1.0 / cutoff) * (3.0 * epsilon) / (2.0 * epsilon + 1.0)
        pair_energies = cutoff_values * weighted_charge_products * (
            1.0 / distances + k_rf * distances**2 - c_rf
        )
        pair_energies = jnp.where(distances < cutoff, pair_energies, 0.0)
    return float(model.coulomb_factor) * pair_energies.sum()


def node_energies_and_charges(
    model,
    positions,
    species,
    edge_src,
    edge_dst,
    pbc_shifts,
    edge_mask,
    total_charge,
):
    weights = model.params
    distances, unit_vectors, radial_features, edge_mask = edge_features(
        positions,
        edge_src,
        edge_dst,
        pbc_shifts,
        edge_mask,
        weights,
        model.cutoff,
        model.cutoff_lower,
        model.alpha,
    )

    tensor_features = embedding(
        species,
        edge_src,
        edge_dst,
        distances,
        unit_vectors,
        radial_features,
        model.cutoff,
        model.cutoff_lower,
        weights["tensor_embedding"],
        edge_mask=edge_mask,
    )

    partial_charges = None
    charge_history = []
    if model.charge_predictors:
        partial_charges = charge_prediction(
            tensor_features,
            weights["charge_predict_0"],
            total_charge,
        )
        charge_history.append(partial_charges)

    for layer_index, layer in enumerate(weights["layers"]):
        tensor_features = message_passing(
            tensor_features,
            partial_charges,
            total_charge,
            edge_src,
            edge_dst,
            distances,
            radial_features,
            layer,
            model.cutoff,
            model.cutoff_lower,
            model.group,
            edge_charge_features=model.edge_charge_features,
            total_charge_interaction_scale=model.total_charge_interaction_scale,
            edge_mask=edge_mask,
        )
        if model.charge_predictors:
            partial_charges = charge_prediction(
                tensor_features,
                weights["charge_predicts"][layer_index],
                total_charge,
            )
            charge_history.append(partial_charges)

    return node_energies(tensor_features, weights), charge_history


class AceFFModel(eqx.Module):
    params: object
    name: str = eqx.field(static=True)
    architecture: str = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    ev_to_kjmol: float = eqx.field(static=True)
    cutoff_lower: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    group: str = eqx.field(static=True)
    charge_predictors: bool = eqx.field(static=True)
    edge_charge_features: bool = eqx.field(static=True)
    total_charge_interaction_scale: bool = eqx.field(static=True)
    coulomb_energy: bool = eqx.field(static=True)
    coulomb_factor: float = eqx.field(static=True)
    coulomb_damp_cutoff: float = eqx.field(static=True)
    coulomb_cutoff: float | None = eqx.field(static=True)
    coulomb_epsilon_solvent: float = eqx.field(static=True)
    exp_minus_1: float = eqx.field(static=True)
    neighbor_cell_atom_threshold: int = eqx.field(static=True)
    neighbor_cell_capacity_multiplier: float = eqx.field(static=True)

    def __init__(self, params, config):
        self.params = params
        self.name = str(config["name"])
        self.architecture = str(config["architecture"])
        self.cutoff = float(config["cutoff"])
        self.ev_to_kjmol = float(config["ev_to_kjmol"])
        self.cutoff_lower = float(config["cutoff_lower"])
        self.alpha = float(config["alpha"])
        self.group = str(config["group"])
        self.charge_predictors = bool(config["charge_predictors"])
        self.edge_charge_features = bool(config["edge_charge_features"])
        self.total_charge_interaction_scale = bool(config["total_charge_interaction_scale"])
        self.coulomb_energy = bool(config["coulomb_energy"])
        self.coulomb_factor = float(config["coulomb_factor"])
        self.coulomb_damp_cutoff = float(config["coulomb_damp_cutoff"])
        coulomb_cutoff = config.get("coulomb_cutoff", None)
        self.coulomb_cutoff = None if coulomb_cutoff is None else float(coulomb_cutoff)
        self.coulomb_epsilon_solvent = float(config.get("coulomb_epsilon_solvent", 78.3))
        self.exp_minus_1 = float(config["exp_minus_1"])
        self.neighbor_cell_atom_threshold = int(config["neighbor_cell_atom_threshold"])
        self.neighbor_cell_capacity_multiplier = float(config["neighbor_cell_capacity_multiplier"])

    def __call__(
        self,
        positions,
        species,
        *,
        box_vectors=None,
        neighbors=None,
        neighbor_idx=None,
        periodic=False,
        total_charge=0.0,
        extra_capacity=0,
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

        edge_src, edge_dst, pbc_shifts, edge_mask = dense_neighbor_edges(
            positions,
            neighbor_idx,
            cutoff=float(self.cutoff),
            box_vectors=box_vectors if periodic else None,
        )
        node_energies, partial_charges = node_energies_and_charges(
            self,
            positions,
            species,
            edge_src,
            edge_dst,
            pbc_shifts,
            edge_mask,
            total_charge,
        )
        local_energy = jnp.sum(node_energies)
        if not self.coulomb_energy:
            return local_energy
        coulomb_energy_term = coulomb_energy(
            self,
            positions,
            partial_charges,
            box_vectors=box_vectors if periodic else None,
        )
        return local_energy + coulomb_energy_term

def load_aceff_model(
    model: str | PathLike = "aceff-jax-2.0",
    *,
    model_path: str | PathLike | None = None,
    dtype: Any = jnp.float32,
    neighbor_cell_atom_threshold: int | None = None,
    neighbor_cell_capacity_multiplier: float | None = None,
):
    if model_path is not None:
        path = Path(model_path)
    elif isinstance(model, PathLike):
        path = Path(model)
    elif model in ACEFF_MODEL_PATHS:
        path = ACEFF_MODEL_PATHS[model]
    else:
        path = Path(model)
    if not path.is_file():
        raise FileNotFoundError(f"AceFF .eqx checkpoint not found: {path}")

    with path.open("rb") as handle:
        config = dict(json.loads(handle.readline().decode()))
        if neighbor_cell_atom_threshold is not None:
            config["neighbor_cell_atom_threshold"] = int(neighbor_cell_atom_threshold)
        if neighbor_cell_capacity_multiplier is not None:
            config["neighbor_cell_capacity_multiplier"] = float(
                neighbor_cell_capacity_multiplier
            )
        loaded = pickle.load(handle)
        if (
            neighbor_cell_atom_threshold is not None
            or neighbor_cell_capacity_multiplier is not None
        ):
            loaded = AceFFModel(loaded.params, config)
    if dtype == jnp.float32:
        return loaded
    return jax.tree_util.tree_map(
        lambda x: x.astype(dtype)
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        loaded,
    )
