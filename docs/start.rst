.. include:: defs.hrst

.. _start:

***************
Getting Started
***************

This section describes how to get started with PFNET in Python. In particular, it covers prerequisites and installation, and provides a simple example that shows how to use this package.

.. _start_prerequisites:

Prerequisites
=============

Before installing the PFNET Python module, the following tools are needed:

* Linux and macOS:

  * C compiler
  * |make|
  * |python| (2 or 3)
  * |pip|
  
* Windows:

  * |anaconda| (for Python 2.7)
  * |cmake| (choose "Add CMake to the system PATH for all users" during installation)
  * |7-zip| (update system path to include the 7z executable, typically in ``C:\Program Files\7-Zip``)
  * |mingwpy| (use ``pip install -i https://pypi.anaconda.org/carlkl/simple mingwpy``)

.. _start_installation:

Installation
============

After the prerequisites for the appropriate operating system have been obtained, the PFNET Python module can be installed by executing the following commands on the terminal or Anaconda prompt::

  pip install numpy cython
  pip install pfnet

To install the module from source, the code can be obtained from `<https://github.com/ttinoco/PFNET.py>`_, and then the following commands can be executed on the terminal or Anaconda prompt from the root directory of the package::

    pip install numpy cython
    python setup.py install

Running the unit tests can be done with::

    pip install nose
    python setup.py build_ext --inplace
    nosetests -s -v

.. _start_example:

Example
=======

As a simple example of how to use the PFNET Python module, consider the task of constructing a power network from a |MATPOWER| power flow file and computing the average bus degree. This can be done as follows::

  >>> import pfnet
  >>> import numpy as np

  >>> net = pfnet.PyParserMAT().parse('ieee14.m')
  
  >>> print(np.average([bus.degree for bus in net.buses])) 
  2.86

In this example, is it assumed that the Python interpreter is started in a directory where the sample case |ieee14| is located.
