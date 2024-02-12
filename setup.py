import os
import sys
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Define your package version directly here
package_version = '0.1.0'

is_called = [
    "_delaunay_class",
]

files = [
    "SMSOceanMeshToolkit/cpp/delaunay_class.cpp",
]

if os.name == "nt":
    home = os.environ.get("USERPROFILE", "").replace("\\", "/")
    vcpkg = f"{home}/SMSOceanMeshToolkit/vcpkg/installed/x64-windows"
    ext_modules = [
        Pybind11Extension(
            loc,
            [fi],
            include_dirs=[f"{vcpkg}/include"],
            extra_link_args=[f"/LIBPATH:{vcpkg}/lib"],
            libraries=["gmp", "mpfr"],
        )
        for fi, loc in zip(files, is_called)
    ]
else:
    ext_modules = [
        Pybind11Extension(loc, [fi], libraries=["gmp", "mpfr"])
        for fi, loc in zip(files, is_called)
    ]

if __name__ == "__main__":
    setup(
        name='SMSOceanMeshToolkit',  # Specify your package name
        version=package_version,
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
    )
