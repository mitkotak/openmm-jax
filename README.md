# OpenMM-JAX

[OpenMM](http://openmm.org) plugin for running JAX force fields through `JaxForce`. For pre-trained machine learning force field models, see [bio-mlff](https://github.com/mitkotak/bio-mlff.git).


```bash
pip install openmmjax-cu12
pip install openmmjax-cu13
```

## Using JaxForce

```python
import jax
import jax.numpy as jnp
from openmmjax import JaxForce
from openmmjax_export import configure_pjrt_plugin, export_jax_model

# Separate functions for energy, forces, energy + forces for computational efficiency.
def compute_energy(positions, box_vectors):
    box_size = jnp.diag(box_vectors)
    wrapped = positions - jnp.floor(positions / box_size) * box_size
    return jnp.sum(wrapped**2)

def compute_forces(positions, box_vectors):
    return -jax.grad(compute_energy)(positions, box_vectors)

def compute_energy_and_forces(positions, box_vectors):
    energy, gradients = jax.value_and_grad(compute_energy)(positions, box_vectors)
    return energy, -gradients

# Apply to particles at indices 0 and 2
selected = [0, 2]

force_mlir, energy_mlir, energy_and_forces_mlir, compile_options = export_jax_model(
    num_model_atoms=len(selected), # needed for shape specialization
    force_function=compute_forces,
    energy_function=compute_energy,
    energy_and_forces_function=compute_energy_and_forces,
    periodic=True,
)

force = JaxForce(force_mlir, energy_mlir, energy_and_forces_mlir, compile_options)

# If PBC is turned off then remove box vectors as input
force.setUsesPeriodicBoundaryConditions(True)
force.setParticles(selected)
configure_pjrt_plugin(force)
```

## Building from Source

```bash
git clone https://github.com/mitkotak/openmm-jax.git
cd openmm-jax
micromamba create -f environment.yml
micromamba activate openmm-jax
cmake -S . -B build \
  -DOPENMM_DIR="$CONDA_PREFIX" \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"
cmake --build build --target install --parallel
cmake --build build --target PythonInstall --parallel
```

## Design Notes

- Most of the frontend is directly borrowed from [openmm-torch](https://github.com/openmm/openmm-torch.git) with the following main changes:
    - `JaxForce` expects separate functions for `energy`, `forces` and `energy + forces` to export instead of relying on a general `energy + forces` function. This saves up compute time when OpenMM requests only energy or forces.
    - Instead of compiling and storing the checkpoints on disk (for e.g. `.pt, .hlo`), the exported functions are converted to `.mlir` strings and then converted to PJRT executables loaded at runtime (`PjrtRuntime::initialize` / `compileStablehloExecutable`). This avoids creating extra files during testing, but support can be added if needed.

- For the backend the key code complexity is in managing memory ownership and stream synchronization in moving from OpenMM to PJRT and back to OpenMM. The OpenMM to PJRT handoff is relatively straightforward since OpenMM owns all the memory until the handoff. To avoid stream syncs or D2D copies, a CUDA event coordinates the input handoff. The PJRT to OpenMM handoff on the other hand is much more complicated since PJRT now owns the memory. The output pointer is extracted from PJRT, followed by launching OpenMM's `addForce` kernel against it which adds the PJRT output to its global state and then deferring releasing the PJRT buffer until the kernel has finished (`OpenMmPjrtOutputLifetime.h/.cpp`). This part of the design was derived from our understanding of `PJRT_Event_Await` so if there's other APIs in the PJRT infrastructure that we missed let us know.

- There's PJRT boilerplate for loading plugins (`PjrtPlugin.h/.cpp`), managing PJRT client sessions (`PjrtClientSession.h/.cpp`), wrapping device-buffer interop (`PjrtBufferInterop.h/.cpp`), and compiling/executing loaded executables (`PjrtLoadedExecutable.h/.cpp`). A lot of this machinery is borrowed from the PJRT C++ API which unfortunately comes with a heavy XLA build if we depend directly in it. This is why we directly copy PJRT's C API (`pjrt_c_api.h`) which is header only. There are also RAII style guards over PJRT handles (`PjrtHandles.h/cpp`) and CUDA contexts (`CudaPrimaryContextGuard.h`)


## Acknowledgements

https://github.com/openmm/openmm/issues/4594 for the idea

@abhijeetgangan for discussions on API design, [openmm-torch](https://github.com/openmm/openmm-torch) and [openmm](https://github.com/openmm/openmm) for the MD API, [xla](https://github.com/openxla/xla/tree/main) for PJRT code, [https://www.youtube.com/watch?v=2GlMqaNxP_w] for intro to PJRT concepts, 

Also show some love to our friends at [lammps-jax](https://github.com/abhijeetgangan/lammps-jax).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
