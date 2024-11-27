.. _api:

API Reference
=============

This page provides documentation for all public-facing objects, methods,  and functions in ``reciprocalspaceship``. Since this
library was designed to provide a "crystallographically-aware" ``pandas.DataFrame``, many methods are inherited directly
from or are compatible with the `Pandas API <https://pandas.pydata.org/docs/reference/index.html#api>`_.

.. Warning::
   Functions available through the ``rs.utils`` namespace are considered private, and may change between releases.
   These functions underlie the symmetry/spacegroup-related operations that are supported in ``reciprocalspaceship``,
   and user-facing methods that use the ``rs.utils`` functions are available as methods of ``rs.DataSet``.

Reflection Data
---------------

Python objects for representing reflection data -- tabular data with values (columns) that are indexed by Miller HKL indices.
These two objects can be thought of as crystallographically-aware versions of ``pandas.DataFrame`` and ``pandas.Series``, respectively.

.. currentmodule:: reciprocalspaceship
.. autosummary::
   :toctree: autoapi
   :nosignatures:
   :template: class.rst

   ~reciprocalspaceship.DataSet
   ~reciprocalspaceship.DataSeries

Input/Output
------------

Supported crystallographic file formats for data I/O. Although they are not listed below, write functions for reflection file formats
are available as methods of ``rs.DataSet`` objects.

.. currentmodule:: reciprocalspaceship
.. autosummary::
   :toctree: autoapi
   :nosignatures:

   ~reciprocalspaceship.read_mtz
   ~reciprocalspaceship.read_cif
   ~reciprocalspaceship.read_csv
   ~reciprocalspaceship.read_pickle
   ~reciprocalspaceship.read_precognition
   ~reciprocalspaceship.read_crystfel
   ~reciprocalspaceship.io.write_ccp4_map

Algorithms
----------

Algorithms for processing reflection data stored in ``rs.DataSet`` objects.

.. currentmodule:: reciprocalspaceship
.. autosummary::
   :toctree: autoapi
   :nosignatures:

   ~reciprocalspaceship.algorithms.merge
   ~reciprocalspaceship.algorithms.scale_merged_intensities
   ~reciprocalspaceship.algorithms.compute_intensity_from_structurefactor

Summary Statistics
------------------

Summary statistics that can be computed from ``rs.DataSet`` objects.

.. currentmodule:: reciprocalspaceship
.. autosummary::
   :toctree: autoapi
   :nosignatures:

   ~reciprocalspaceship.stats.compute_completeness

Helpful Functions
-----------------

Functions intended to help users work with the library. These functions are available from the top-level ``rs`` namespace.

.. currentmodule:: reciprocalspaceship
.. autosummary::
   :toctree: autoapi
   :nosignatures:

   ~reciprocalspaceship.concat
   ~reciprocalspaceship.summarize_mtz_dtypes

Commandline Tools
-----------------

Commandline tools to support common use cases.

.. toctree::
   :maxdepth: 1
   :hidden:

   mtzdump
   cifdump

.. Manual table for commandline tools

.. raw:: html

   <table class="longtable docutils align-default">
	 <colgroup>
	 <col style="width: 10%" />
	 <col style="width: 90%" />
	 </colgroup>
	 <tbody>
	 <tr class="row-odd"><td><p><a class="reference internal" href="mtzdump.html"><code class="xref py py-obj docutils literal notranslate"><span class="pre">rs.mtzdump</span></code></a></p></td>
	 <td><p>Summarize the contents of an MTZ file</p></td>
	 </tr>
    <tr class="row-even"><td><p><a class="reference internal" href="cifdump.html"><code class="xref py py-obj docutils literal notranslate"><span class="pre">rs.cifdump</span></code></a></p></td>
    <td><p>Summarize the contents of a CIF file</p></td>
    </tr>
	 </tbody>
   </table>
