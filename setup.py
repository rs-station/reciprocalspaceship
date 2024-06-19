from setuptools import find_packages, setup


# Get version number
def getVersionNumber():
    with open("reciprocalspaceship/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version


__version__ = getVersionNumber()

DESCRIPTION = "Tools for exploring reciprocal space"
LONG_DESCRIPTION = """
``reciprocalspaceship`` provides a ``pandas``-style interface for
analyzing and manipulating reflection data from crystallography
experiments. Using this library, it is possible to interactively work
with crystallographic data in Python, enabling easy integration  with
modern scientific computing libraries. ``reciprocalspaceship`` is
intended to support the rapid prototyping of new crystallographic methods
and custom analyses while maintaining clear, reproducible, and performant
code.

Features of this library include:

  - Crystallographically-aware ``pandas`` objects, datatypes, and syntax
    that are familiar to Python users.
  - Convenient integration with `GEMMI <https://gemmi.readthedocs.io/en/latest/>`__
    to provide built-in methods and support for developing functions that
    use space groups, unit cell parameters, and crystallographic symmetry
    operations.
  - Support for reading and writing MTZ reflection files.
"""
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/rs-station/reciprocalspaceship/issues",
    "Documentation": "https://rs-station.github.io/reciprocalspaceship/",
    "Source Code": "https://github.com/rs-station/reciprocalspaceship",
}

# Testing requirements
tests_require = ["pytest", "pytest-cov", "pytest-xdist", "ray"]

# Documentation requirements
docs_require = [
    "sphinx",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx-design",
    "sphinxcontrib-autoprogram",
    "autodocsumm",
]

# Examples requirements
examples_require = [
    "jupyter",
    "tqdm",
    "matplotlib",
    "seaborn",
    "celluloid",
    "scikit-image",
]

setup(
    name="reciprocalspaceship",
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    license="MIT",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Kevin M. Dalton, Jack B. Greisman",
    author_email="kmdalton@g.harvard.edu, greisman@g.harvard.edu",
    url="https://rs-station.github.io/reciprocalspaceship/",
    project_urls=PROJECT_URLS,
    python_requires=">=3.9",
    install_requires=[
        "gemmi>=0.5.5, <=0.6.6",
        "pandas>=2.2.2, <=2.2.2",
        "numpy",
        "scipy",
        "ipython",
    ],
    setup_requires=["pytest-runner"],
    tests_require=tests_require,
    extras_require={
        "dev": tests_require + docs_require + examples_require,
        "examples": examples_require,
    },
    entry_points={
        "console_scripts": [
            "rs.mtzdump=reciprocalspaceship.commandline.mtzdump:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python",
    ],
)
