from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

version = "@OPENMM_JAX_VERSION@"
openmm_dir = "@OPENMM_DIR@"
jax_plugin_header_dir = "@JAX_PLUGIN_HEADER_DIR@"
jax_plugin_library_dir = "@JAX_PLUGIN_LIBRARY_DIR@"
package_name = os.environ.get("OPENMM_JAX_PACKAGE_NAME", "openmmjax")
cuda_version = os.environ.get("OPENMM_JAX_CUDA_VERSION")
readme = Path(__file__).with_name("README.md")

if cuda_version not in {None, "", "12", "13"}:
    raise RuntimeError("OPENMM_JAX_CUDA_VERSION must be unset, '12', or '13'")

extra_compile_args = ["-std=c++17"]
extra_link_args = []
runtime_library_dirs = [
    os.path.join(openmm_dir, "lib"),
    jax_plugin_library_dir,
]

if platform.system() == "Windows":
    extra_compile_args = ["/std:c++17"]
    runtime_library_dirs = None
elif platform.system() == "Darwin":
    extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.13"]
    extra_link_args += ["-stdlib=libc++", "-mmacosx-version-min=10.13"]
elif platform.system() == "Linux":
    extra_link_args += ["-Wl,--enable-new-dtags", "-Wl,-rpath,$ORIGIN"]
    runtime_library_dirs += ["$ORIGIN"]

extension = Extension(
    name="_openmmjax",
    sources=["JaxPluginWrapper.cpp"],
    libraries=["OpenMM", "OpenMMJax"],
    include_dirs=[
        os.path.join(openmm_dir, "include"),
        jax_plugin_header_dir,
    ],
    library_dirs=[
        os.path.join(openmm_dir, "lib"),
        jax_plugin_library_dir,
    ],
    runtime_library_dirs=runtime_library_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)


class BundleBuildExt(build_ext):
    """Copy native OpenMM-JAX libraries beside the Python extension for wheels."""

    def run(self):
        self._bundled_libraries: list[str] = []
        super().run()
        ext_dir = Path(self.get_ext_fullpath("_openmmjax")).resolve().parent
        for library_dir in self._runtime_library_dirs():
            if not library_dir.is_dir():
                continue
            for library in self._iter_openmmjax_libraries(library_dir):
                target = ext_dir / library.name
                if library.resolve() == target.resolve():
                    continue
                shutil.copy2(library, target)
                self._bundled_libraries.append(str(target))

    def get_outputs(self):
        return super().get_outputs() + getattr(self, "_bundled_libraries", [])

    @staticmethod
    def _runtime_library_dirs() -> tuple[Path, ...]:
        return (
            Path(jax_plugin_library_dir),
            Path(jax_plugin_library_dir) / "plugins",
        )

    @staticmethod
    def _iter_openmmjax_libraries(directory: Path):
        system = platform.system()
        if system == "Windows":
            patterns = ("OpenMMJax*.dll",)
        elif system == "Darwin":
            patterns = ("libOpenMMJax*.dylib",)
        else:
            patterns = ("libOpenMMJax*.so", "libOpenMMJax*.so.*")
        seen: set[Path] = set()
        for pattern in patterns:
            for library in sorted(directory.glob(pattern)):
                resolved = library.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield library


setup(
    name=package_name,
    version=version,
    description="OpenMM plugin for exported JAX/XLA models",
    long_description=readme.read_text(encoding="utf-8") if readme.is_file() else None,
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=["LICENSE"],
    url="https://github.com/mitkotak/openmm-jax",
    py_modules=["openmmjax", "openmmjax_export"],
    ext_modules=[extension],
    cmdclass={"build_ext": BundleBuildExt},
    install_requires=[
        "openmm>=8.5.2",
        f"jax[cuda{cuda_version}]" if cuda_version in {"12", "13"} else "jax",
        "numpy",
    ],
    python_requires=">=3.11",
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
