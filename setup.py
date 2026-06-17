"""Python packaging for OpenMM-JAX.

Build and install the native OpenMM plugins with CMake before invoking pip.
This setup script builds the Python extension from the SWIG wrapper generated
by CMake.
"""

from __future__ import annotations

import importlib
import os
import platform
import re
from pathlib import Path

from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent
CMAKE_VERSION_RE = re.compile(
    r"SET\(OPENMM_JAX_VERSION\s+([0-9.]+)",
    re.IGNORECASE,
)


def read_version() -> str:
    cmake_lists = ROOT / "CMakeLists.txt"
    match = CMAKE_VERSION_RE.search(cmake_lists.read_text(encoding="utf-8"))
    if match:
        return match.group(1)
    return "0.1"


def openmm_dir() -> Path:
    value = os.environ.get("OPENMM_DIR")
    if value:
        return Path(value).resolve()

    try:
        openmm_module = importlib.import_module("openmm")
        return Path(openmm_module.version.openmm_library_path).resolve().parent
    except ImportError:
        pass

    value = os.environ.get("CONDA_PREFIX")
    if value:
        return Path(value).resolve()
    raise RuntimeError(
        "Set OPENMM_DIR to the OpenMM installation prefix, or install openmm "
        "in the active Python environment"
    )


def make_extension() -> Extension:
    openmm = openmm_dir()

    extra_compile_args = ["-std=c++17"]
    extra_link_args: list[str] = []
    if platform.system() == "Windows":
        extra_compile_args = ["/std:c++17"]
    elif platform.system() == "Darwin":
        extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.13"]
        extra_link_args += ["-stdlib=libc++", "-mmacosx-version-min=10.13"]

    library_dirs = [str(openmm / "lib")]
    runtime_library_dirs = [str(openmm / "lib")]
    if platform.system() == "Linux":
        extra_link_args += [
            "-Wl,--enable-new-dtags",
            f"-Wl,-rpath,{openmm / 'lib'}",
        ]
    if platform.system() == "Windows":
        runtime_library_dirs = None

    return Extension(
        name="_openmmjax",
        sources=["python/JaxPluginWrapper.cpp"],
        libraries=["OpenMM", "OpenMMJax"],
        include_dirs=[str(openmm / "include"), str(ROOT / "openmmapi" / "include")],
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


if __name__ == "__main__":
    setup(
        name="openmmjax",
        version=read_version(),
        description="OpenMM plugin for exported JAX/XLA models",
        license="MIT",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.11",
        ],
        packages=["openmmjax_models"],
        package_dir={"": "python", "openmmjax_models": "models"},
        package_data={"openmmjax_models": ["ani2x_model0.eqx"]},
        py_modules=["openmmjax", "openmmjax_export"],
        ext_modules=[make_extension()],
        install_requires=[
            "openmm>=8.2",
            "openmmml>=1.2",
            "equinox>=0.13.8",
            "jax",
            "jax-md>=0.2.29",
            "jaxlib",
            "numpy",
        ],
        extras_require={
            "lint": [
                "ruff>=0.12",
            ],
        },
        python_requires=">=3.11",
    )
