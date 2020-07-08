.. _overview:

Overview and Goals
==================

Much of crystallographic data analysis is handled using a series of software suites and commandline
tools. This practice has greatly simplified the routine processing of diffraction images to determine
new structures. However, it has made it difficult to develop new crystallographic methods and
analyses. In order to develop a custom method or analysis it is necessary to familiarize oneself
with a large library that can be difficult to integrate with other existing tools and libraries.

The goal of ``reciprocalspaceship`` is to make it easy to work with crystallographic data in Python.
Reflection data from a diffraction experiment is inherently tabular -- integrated intensities with
associated HKL indices and metadata -- and can be represented well in a ``pandas.DataFrame``. Our
library extends this idea further by making these ``DataFrames`` crystallographically-aware.

By calling ``import reciprocalspaceship as rs`` you get access to ``rs.DataSet``, which enables the
use of HKL indices, space groups, unit cell parameters, and symmetry operations to manipulate
reflection data.
Many common routines, such as applying symmetry operations or mapping reflections to the reciprocal
space asymmetric unit are already implemented as built-in methods. Perhaps most importantly,
``rs.DataSet`` objects make it easy to use modern scientific computing libraries to support new
methods and analyses, and can be used to write new analysis pipelines in a convenient and performant
way without sacrificing reproducibility.


