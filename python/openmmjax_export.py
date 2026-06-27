from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp


def configure_pjrt_plugin(
    force: Any,
    plugin: str | None = None,
    plugin_modules: tuple[str, ...] = (
        "jax_plugins.xla_cuda13",
        "jax_plugins.xla_cuda12",
    ),
    plugin_library: str = "xla_cuda_plugin.so",
) -> str:
    """Resolve and attach a PJRT plugin to a JaxForce instance."""
    plugin = plugin or force.getPjrtPluginPath() or None
    module_names = (plugin,) if plugin is not None else plugin_modules
    if plugin is not None:
        path = Path(plugin).expanduser().resolve()
        if path.is_file():
            force.setPjrtPluginPath(str(path))
            return str(path)

    for module_name in module_names:
        try:
            spec = importlib.util.find_spec(module_name)
        except (ModuleNotFoundError, ValueError):
            spec = None
        if spec is not None and spec.origin is not None:
            path = Path(spec.origin).resolve().parent / plugin_library
            if path.is_file():
                force.setPjrtPluginPath(str(path))
                return str(path)

    if plugin is not None:
        raise FileNotFoundError(f"Could not resolve CUDA PJRT plugin '{plugin}'.")
    modules = ", ".join(plugin_modules)
    raise FileNotFoundError(
        f"Could not find {plugin_library}. Install a JAX CUDA PJRT "
        f"package such as one of: {modules}, or pass an explicit plugin path."
    )


def get_compile_options(
    num_replicas: int = 1,
    num_partitions: int = 1,
) -> str:
    """Return serialized XLA compile options encoded safely for SWIG strings."""
    from jaxlib import xla_client

    options = xla_client.CompileOptions()
    options.num_replicas = num_replicas
    options.num_partitions = num_partitions
    return base64.b64encode(options.SerializeAsString()).decode("ascii")


def export_jax_model(
    *,
    num_system_atoms: int,
    force_function: Callable[..., Any],
    energy_function: Callable[..., Any],
    energy_and_forces_function: Callable[..., Any],
    periodic: bool,
    input_dtype: Any = jnp.float32,
) -> tuple[str, str, str, str]:
    """Export OpenMM-shaped JAX callables for native JaxForce execution."""
    platforms = ("cuda",)

    positions_shape = jax.ShapeDtypeStruct((int(num_system_atoms), 3), input_dtype)
    export_args: tuple[Any, ...]
    if periodic:
        box_vectors_shape = jax.ShapeDtypeStruct((3, 3), input_dtype)
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
    compile_options_base64 = get_compile_options()

    return force_mlir, energy_mlir, energy_and_forces_mlir, compile_options_base64
