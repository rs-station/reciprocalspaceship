.. _documentation:

Building Documentation
======================

The ``reciprocalspaceship`` documentation is built using `sphinx <http://sphinx-doc.org/>`_. There are a few additional dependencies that can all be installed using ``pip`` or ``conda``::

  conda install sphinx sphinx_rtd_theme

The documentation can then be built using the ``Makefile`` in the ``docs/`` subdirectory::

  cd docs
  make html
  cd ..

The ``index.html`` of the documentation will then be found in ``docs/_build/``.

If you have a partial build of the documentation, or if you make a change to the reStructuredText (RST) templates, it may be necessary to remove the existing documentation files. This can be accomplished by running the following command in the ``docs/`` subdirectory::

  make clean


