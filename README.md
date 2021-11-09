# 1 /:rocket: = reciprocalspaceship
![Build](https://github.com/Hekstra-Lab/reciprocalspaceship/workflows/Build/badge.svg)
[![Documentation](https://github.com/Hekstra-Lab/reciprocalspaceship/workflows/Documentation/badge.svg)](https://hekstra-lab.github.io/reciprocalspaceship)
[![PyPI](https://img.shields.io/pypi/v/reciprocalspaceship?color=blue)](https://pypi.org/project/reciprocalspaceship/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/reciprocalspaceship/badges/version.svg)](https://anaconda.org/conda-forge/reciprocalspaceship)
[![codecov](https://codecov.io/gh/Hekstra-Lab/reciprocalspaceship/branch/main/graph/badge.svg)](https://codecov.io/gh/Hekstra-Lab/reciprocalspaceship)
[![apm](https://img.shields.io/apm/l/vim-mode.svg)](https://github.com/Hekstra-Lab/reciprocalspaceship/blob/main/LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hekstra-Lab/reciprocalspaceship/main?filepath=docs%2Fexamples)
[![DOI](https://img.shields.io/badge/DOI-10.1107%2FS160057672100755X-blue)](https://doi.org/10.1107/S160057672100755X)

Tools for exploring reciprocal space.

`reciprocalspaceship` provides a `pandas`-style interface for
analyzing and manipulating reflection data from crystallography
experiments. Using this library, it is possible to interactively
work with crystallographic data in Python, enabling easy
integration with modern scientific computing libraries. `reciprocalspaceship`
is intended to support the rapid prototyping of new crystallographic methods and
custom analyses while maintaining clear and performant code.

Features of this library include:

- Crystallographically-aware `pandas` objects, datatypes, and syntax that are familiar to Python users.
- Convenient integration with [GEMMI](https://gemmi.readthedocs.io/en/latest/) to provide built-in methods and
  support for developing functions that use space groups, unit cell parameters, and crystallographic
  symmetry operations.
- Support for reading and writing MTZ reflection files.

## Installation

The fastest way to install `reciprocalspaceship` is using `pip`:

```
pip install reciprocalspaceship
```

or using `conda`:

```
conda install -c conda-forge reciprocalspaceship
```

For more installation information, see our [installation guide](https://hekstra-lab.github.io/reciprocalspaceship/userguide/installation.html).

## Quickstart

To get started with `reciprocalspaceship`, see our [quickstart guide](https://hekstra-lab.github.io/reciprocalspaceship/examples/quickstart.html).

## Documentation

For more details on the use of `reciprocalspaceship`, check out our [documentation](https://hekstra-lab.github.io/reciprocalspaceship).

## Reference

Our paper describing the library was published in the Journal of Applied Crystallography: [link to paper](https://doi.org/10.1107/S160057672100755X).
Please cite this publication for any work that makes use of this library.

We also have a pre-print describing the library on [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.02.03.429617v1).
