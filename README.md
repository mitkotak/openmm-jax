# OpenMM-JAX

JAX models in OpenMM with `jax.export` + PJRT runtime

```bash
git clone https://github.com/mitkotak/openmm-jax.git
cd openmm-jax
micromamba create -f environment.yml
micromamba activate openmm-jax
cmake -S . -B build \
  -DOPENMM_DIR="$CONDA_PREFIX" \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"
cmake --build build --target install --parallel
cmake --build build --target PythonInstall
```
## Supported Models

- ANI2x (single)
- FeNNix
- AIMNet2
- MACE

## Design Notes

- Most of the frontend is directly borrowed from [openmm-torch](`https://github.com/openmm/openmm-torch.git`) with the following main changes:
    - `JaxForce` expects separate functions for `energy`, `forces` and `energy + forces` to export instead of relying on a general `energy + forces` function. This saves up compute time when OpenMM requests only energy or forces.
    - Instead of compiling and storing the checkpoints on disk (for e.g. `.pt, .hlo`), the exported functions are converted to `.mlir` strings and then convergted to PJRT executables loaded at runtime (`PjrtRuntime::initialize` / `compileStablehloExecutable`). This was to avoid creating too many files during testing but support can be added easily if needed.

- For the backend the key code complexity is in managing memory ownership and stream synchronization in moving from OpenMM to PJRT and back to OpenMM. The OpenMM to PJRT handoff is relatively straightforward since OpenMM owns all the memory until the handoff. To avoid stream syncs or D2D copies, a CUDA event coordinates the input handoff. The PJRT to OpenMM handoff on the other hand is much more complicated since PJRT now owns the memory. The output pointer is extracted from PJRT, followed by launching OpenMM's `addForce` kernel against it which adds the PJRT output to its global state and then deferring releasing the PJRT buffer until the kernel has finished (`OpenMmPjrtOutputLifetime.h/.cpp`). This part of the design was derived from our understanding of `PJRT_Event_Await` so if there's other APIs in the PJRT infrastructure that we missed let us know.

- There's PJRT boilerplate for loading plugins (`PjrtPlugin.h/.cpp`), managing PJRT client sessions (`PjrtClientSession.h/.cpp`), wrapping device-buffer interop (`PjrtBufferInterop.h/.cpp`), and compiling/executing loaded executables (`PjrtLoadedExecutable.h/.cpp`). A lot of this machinery is borrowed from the PJRT C++ API which unfortunately comes with a heavy XLA build if we depend directly in it. This is why we directly copy PJRT's C API (`pjrt_c_api.h`) which is header only. There are also RAII style guards over PJRT handles (`PjrtHandles.h/cpp`) and CUDA contexts (`CudaPrimaryContextGuard.h`)


## Acknowledgements

https://github.com/openmm/openmm/issues/4594 for the idea

@abhijeetgangan for discussions on API design, [openmm-torch](https://github.com/openmm/openmm-torch)/[opemmm-ml](https://github.com/openmm/openmm-ml)/[openmm](https://github.com/openmm/openmm) for MD code and [xla](https://github.com/openxla/xla/tree/main) for PJRT code, [https://www.youtube.com/watch?v=2GlMqaNxP_w] for intro to PJRT concepts, [FeNNol](https://github.com/FeNNol-tools/FeNNol) for their ANI implementation.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
