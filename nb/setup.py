"""Builds and installs the Cython extension modules for the project.

This setup script compiles the Cython `.pyx` files into C extensions, providing
a significant performance boost for computationally intensive operations. It uses
`setuptools` to manage the build process and correctly places the compiled
modules within the package structure.

The primary extension built by this script is `fast_processor`, which is
critical for the context engine's performance.

Attributes:
    extensions (list): A list of `setuptools.Extension` objects, each defining a
        Cython module to be compiled.
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension module explicitly
extensions = [
    Extension(
        # This defines the import path
        name="nb.src.context_engine.fast_processor", 
        # This is the source .pyx file
        sources=["src/context_engine/fast_processor.pyx"],
        # Include numpy headers, good practice for numerical code
        include_dirs=[numpy.get_include()]
    )
]

setup(
    ext_modules=cythonize(extensions),
    zip_safe=False,
) 