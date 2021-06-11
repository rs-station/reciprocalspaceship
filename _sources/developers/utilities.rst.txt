.. _utilities:

Utility Functions
=================

``rs.utils`` contains symmetry/spacegroup-related operations that serve as the foundation for much
of the crystallographic support provided by ``reciprocalspaceship``. These methods are not
considered to be user-facing, but they are documented here because they may be useful for
composing new methods or algorithms.

.. Note::
   In most cases, these functions are written to operate on and return ``NumPy`` arrays. This was
   chosen to ensure that these methods are performant, and can be largely independent of any
   interface decisions related to ``rs.DataSet`` or ``pandas``.

.. automodule:: reciprocalspaceship.utils
   :autosummary:
