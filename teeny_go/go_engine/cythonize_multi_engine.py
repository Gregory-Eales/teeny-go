from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy



setup(
    ext_modules = cythonize("cython_multi_go_engine.pyx"),
    include_dirs=[numpy.get_include()]
)

setup(
    ext_modules=[
        Extension("cython_multi_go_engine", ["cython_multi_go_engine.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)
