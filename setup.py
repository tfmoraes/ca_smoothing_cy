from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cy_mesh", ["cy_mesh.pyx"],
                             include_dirs =  [np.get_include()],
                             extra_compile_args=['-fopenmp', '-std=c++11'],
                             extra_link_args=['-fopenmp', '-std=c++11'],
                             language='c++',),
                   ]

)
