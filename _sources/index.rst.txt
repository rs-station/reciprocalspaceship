ReciprocalSpaceship
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

.. panels::
    :container: container-lg pb-3
    :column: col-lg-4 col-md-4 col-sm-6 col-xs-12 p-2

    The user guide provides information about installing and using ``reciprocalspaceship``
    +++
    .. link-button:: userguide/index
       :type: ref
       :text: User Guide
       :classes: btn-outline-primary btn-block stretched-link
    ---
    The API reference provides detailed information about the Python API for the library
    +++
    .. link-button:: api/index
        :type: ref
        :text: API Reference
        :classes: btn-outline-primary btn-block stretched-link
    ---
    The developer's guide is a resource for those looking to contribute to this project
    +++
    .. link-button:: developers/index
        :type: ref
        :text: Developer Guide
        :classes: btn-outline-primary btn-block stretched-link

.. toctree::
   :maxdepth: 1
   :hidden:

   userguide/index
   api/index
   developers/index
