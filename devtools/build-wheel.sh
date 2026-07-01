#!/usr/bin/env bash
set -euo pipefail

: "${OPENMM_JAX_PACKAGE_NAME:?Set OPENMM_JAX_PACKAGE_NAME}"
: "${OPENMM_JAX_CUDA_VERSION:?Set OPENMM_JAX_CUDA_VERSION}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

python -m pip wheel ./build/python --no-deps -w wheelhouse

auditwheel repair wheelhouse/*.whl -w dist \
  --exclude libOpenMM.so \
  --exclude libOpenMMCUDA.so \
  --exclude libcuda.so.1 \
  --exclude libcuda.so
