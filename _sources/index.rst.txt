reciprocalspaceship
===================

Tools for exploring reciprocal space.

``reciprocalspaceship`` provides a ``pandas``-style interface for
analyzing and manipulating reflection data from crystallography
experiments. Using this library, it is possible to interactively
work with crystallographic data in Python, enabling easy
integration with modern scientific computing libraries. ``reciprocalspaceship``
is intended to support the rapid prototyping of new crystallographic methods and
custom analyses while maintaining clear and performant code.

Features of this library include:

- Crystallographically-aware ``pandas`` objects, datatypes, and syntax that are familiar to Python users.
- Convenient integration with `GEMMI <https://gemmi.readthedocs.io/en/latest/>`_ to provide built-in methods and
  support for developing functions that use space groups, unit cell parameters, and crystallographic
  symmetry operations.
- Support for reading and writing MTZ reflection files.

.. grid:: 3

    .. grid-item-card:: User Guide

       The user guide provides information about installing and using ``reciprocalspaceship``

       .. button-ref:: userguide/index
           :color: primary
	   :shadow:
           :expand:

           User Guide

    .. grid-item-card:: API Reference

        The API reference provides details about the Python API for the library

        .. button-ref:: api/index
           :color: primary
	   :shadow:
           :expand:

           API Reference

    .. grid-item-card:: Developer's Guide

        The developer's guide is a resource for those looking to contribute to this project

	.. button-ref:: developers/index
           :color: primary
	   :shadow:
           :expand:

           Developer's Guide

.. toctree::
   :maxdepth: 1
   :hidden:

   userguide/index
   api/index
   developers/index
