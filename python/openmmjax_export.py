"""Python export helpers for OpenMM-JAX PJRT forces."""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

_PJRT_PLUGIN_MODULES = (
    "jax_plugins.xla_cuda13",
    "jax_plugins.xla_cuda12",
)
_PJRT_PLUGIN_LIBRARY = "xla_cuda_plugin.so"


def _plugin_path_from_module(module_name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ModuleNotFoundError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None

    path = Path(spec.origin).resolve().parent / _PJRT_PLUGIN_LIBRARY
    return str(path) if path.is_file() else None


def get_plugin_path(required: bool = True, *, plugin: str | None = None) -> str | None:
    """Return a loadable CUDA PJRT plugin path.

    ``plugin`` may be an explicit ``.so`` path or a ``jax_plugins.xla_cuda*``
    module name.  If it is omitted, the installed JAX CUDA plugin is detected.
    """
    if plugin:
        path = Path(plugin).expanduser()
        if path.is_file():
            return str(path.resolve())

        module_path = _plugin_path_from_module(plugin)
        if module_path is not None:
            return module_path

        if required:
            raise FileNotFoundError(f"Could not resolve CUDA PJRT plugin '{plugin}'.")
        return None

    for module_name in _PJRT_PLUGIN_MODULES:
        module_path = _plugin_path_from_module(module_name)
        if module_path is not None:
            return module_path

    if required:
        modules = ", ".join(_PJRT_PLUGIN_MODULES)
        raise FileNotFoundError(
            f"Could not find {_PJRT_PLUGIN_LIBRARY}. Install a JAX CUDA PJRT "
            f"package such as one of: {modules}, or pass an explicit plugin path."
        )
    return None


def configure_pjrt_plugin(force: Any, plugin: str | None = None) -> str:
    """Resolve and attach a PJRT plugin to a JaxForce instance."""
    if plugin is None and hasattr(force, "getPjrtPluginPath"):
        existing = force.getPjrtPluginPath()
        if existing:
            plugin = existing
    path = get_plugin_path(plugin=plugin, required=True)
    if path is None:
        raise FileNotFoundError("Could not resolve a CUDA PJRT plugin.")
    force.setPjrtPluginPath(path)
    return path


def get_compile_options(
    num_replicas: int = 1,
    num_partitions: int = 1,
) -> bytes:
    """Return serialized XLA compile options for PJRT C API compilation."""
    from jaxlib import xla_client

    options = xla_client.CompileOptions()
    options.num_replicas = num_replicas
    options.num_partitions = num_partitions
    return options.SerializeAsString()


def get_compile_options_base64(
    num_replicas: int = 1,
    num_partitions: int = 1,
) -> str:
    """Return serialized XLA compile options encoded safely for SWIG strings."""
    return base64.b64encode(get_compile_options(num_replicas, num_partitions)).decode("ascii")


def export_jax_model(
    *,
    num_system_atoms: int,
    force_function: Callable[..., Any],
    energy_function: Callable[..., Any],
    energy_and_forces_function: Callable[..., tuple[Any, Any]],
    periodic: bool,
) -> tuple[str, str, str, str]:
    """Export full-system OpenMM-shaped JAX callables for JaxForce."""
    platforms = ("cuda",)

    positions_shape = jax.ShapeDtypeStruct((int(num_system_atoms), 3), jnp.float32)
    export_args: tuple[Any, ...]
    if periodic:
        box_vectors_shape = jax.ShapeDtypeStruct((3, 3), jnp.float32)
        export_args = positions_shape, box_vectors_shape
    else:
        export_args = (positions_shape,)

    def export_stablehlo(function: Callable[..., Any]) -> str:
        exported = jax.export.export(
            jax.jit(function),
            platforms=platforms,
        )(*export_args)
        return exported.mlir_module()

    force_mlir = export_stablehlo(force_function)
    energy_mlir = export_stablehlo(energy_function)
    energy_and_forces_mlir = export_stablehlo(energy_and_forces_function)
    compile_options_base64 = get_compile_options_base64()

    return force_mlir, energy_mlir, energy_and_forces_mlir, compile_options_base64
