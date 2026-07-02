#!/usr/bin/env bash
set -euo pipefail

: "${OPENMM_JAX_PACKAGE_NAME:?Set OPENMM_JAX_PACKAGE_NAME}"
: "${OPENMM_JAX_CUDA_VERSION:?Set OPENMM_JAX_CUDA_VERSION}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

rm -rf wheelhouse dist
mkdir -p wheelhouse dist

openmm_jax_rpath='$ORIGIN:$ORIGIN/OpenMM.libs/lib:$ORIGIN/../../..'
openmm_jax_cuda_rpath='$ORIGIN:$ORIGIN/..:$ORIGIN/OpenMM.libs/lib:$ORIGIN/OpenMM.libs/lib/plugins:$ORIGIN/../../..:$ORIGIN/../../../plugins'
openmm_jax_python_rpath='$ORIGIN:$ORIGIN/OpenMM.libs/lib:$ORIGIN/OpenMM.libs/lib/plugins:$ORIGIN/../../..:$ORIGIN/../../../plugins'

patchelf --set-rpath "${openmm_jax_rpath}" build/libOpenMMJax.so
patchelf --set-rpath "${openmm_jax_cuda_rpath}" build/libOpenMMJaxCUDA.so

python -m pip wheel ./build/python --no-deps -w wheelhouse

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT
python -m wheel unpack wheelhouse/*.whl --dest "${tmpdir}"
wheel_dir=("${tmpdir}"/*)
patchelf --set-rpath "${openmm_jax_python_rpath}" "${wheel_dir[0]}"/_openmmjax*.so
patchelf --set-rpath "${openmm_jax_rpath}" "${wheel_dir[0]}"/libOpenMMJax.so
patchelf --set-rpath "${openmm_jax_cuda_rpath}" "${wheel_dir[0]}"/libOpenMMJaxCUDA.so
rm -f wheelhouse/*.whl
python -m wheel pack "${wheel_dir[0]}" --dest-dir wheelhouse

auditwheel repair wheelhouse/*.whl -w dist \
  --exclude libOpenMMJax.so \
  --exclude libOpenMM.so \
  --exclude libOpenMMCUDA.so \
  --exclude libcuda.so.1 \
  --exclude libcuda.so
