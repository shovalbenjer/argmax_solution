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
from setuptools import setup, find_packages

setup(
    name="recipe_analysis",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where='src'),
) 