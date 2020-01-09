.. _installation:

Installation
============

``reciprocalspaceship`` is currently only available as a private repository. For installation,
we recommend cloning the repository and installing the package to a ``conda`` environment::

  conda install pybind11
  git clone https://github.com/Hekstra-Lab/reciprocalspaceship
  cd reciprocalspaceship
  python setup.py install

Notes:

- It is necessary to install ``pybind11`` before running the ``setup.py`` because of a
  dependency in ``gemmi``. This will hopefully be fixed in the near future.  
- These installation instructions should be updated when ``reciprocalspaceship`` is available
  through PyPI and can be installed with ``pip`` or ``conda``

