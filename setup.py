#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import os
import sys
import numpy as np
from subprocess import call
from Cython.Build import cythonize
from setuptools import setup, Extension

# C library build
if 'darwin' in sys.platform.lower() or 'linux' in sys.platform.lower():
    return_code = call(["./build_lib.sh"])
else:
    return_code = call(["build_lib.bat"])
if return_code != 0:
    raise ValueError('Unable to build C library')

# Extra link args
if 'darwin' in sys.platform.lower():
    extra_link_args=['-Wl,-rpath,@loader_path/']
elif 'linux' in sys.platform.lower():
    extra_link_args=['-Wl,-rpath=$ORIGIN']
else:
    extra_link_args=['']

exec(open(os.path.join('pfnet', 'version.py')).read())

setup(name='PFNET',
      zip_safe=False,
      version=__version__,
      description='Power Flow Network Library',
      url='https://github.com/ttinoco/PFNET',
      author='Tomas Tinoco De Rubira',
      author_email='ttinoco5687@gmail.com',
      include_package_data=True,
      license='BSD 2-Clause License',
      packages=['pfnet',
                'pfnet.functions',
                'pfnet.constraints',
                'pfnet.parsers',
                'pfnet.tests',
                'pfnet.utils'],
      install_requires=['cython>=0.20.1',
                        'numpy>=1.11.2',
                        'scipy>=0.18.1',
                        'grg_mpdata>=0.1.0',
                        'grg_pssedata>=0.1.0',
                        'nose'],
      package_data={'pfnet':['libpfnet*', '*.dll']},
      classifiers=['Development Status :: 5 - Production/Stable',
                   'License :: OSI Approved :: BSD License',
                   'Programming Language :: Python :: 3.6'],
      ext_modules=cythonize([Extension(name="pfnet.cpfnet",
                                       sources=["./pfnet/cpfnet.pyx"],
                                       libraries=['pfnet'],
                                       include_dirs=[np.get_include(),'./lib/pfnet/include'],
                                       library_dirs=['./pfnet'],
                                       extra_link_args=extra_link_args)]))
                                       
