from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("pf_engine_cpy.pyx")
)