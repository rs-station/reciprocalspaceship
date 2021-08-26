from setuptools import setup, find_packages

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
    "Bug Tracker": "https://github.com/Hekstra-Lab/reciprocalspaceship/issues",
    "Documentation": "https://hekstra-lab.github.io/reciprocalspaceship/",
    "Source Code": "https://github.com/Hekstra-Lab/reciprocalspaceship",
}

# Testing requirements
tests_require = ["pytest", "pytest-cov", "pytest-xdist"]

# Documentation requirements
docs_require = [
    # sphinx documentation
    "sphinx",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx-panels",
    "sphinxcontrib-autoprogram",
    "jupyter",
    "autodocsumm",
    # example notebooks
    "tqdm",
    "matplotlib",
    "seaborn",
    "celluloid",
    "scikit-image",
    "torch",
]

setup(
    name="reciprocalspaceship",
    packages=find_packages(),
    version=__version__,
    license="MIT",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Kevin M. Dalton, Jack B. Greisman",
    author_email="kmdalton@g.harvard.edu, greisman@g.harvard.edu",
    url="https://hekstra-lab.github.io/reciprocalspaceship/",
    project_urls=PROJECT_URLS,
    python_requires=">3.6",
    install_requires=[
        "gemmi >= 0.4.2",
        "pandas >= 1.2.0, <= 1.3.2",
        "numpy",
        "scipy",
        "ipython",
    ],
    setup_requires=["pytest-runner"],
    tests_require=tests_require,
    extras_require={"dev": tests_require + docs_require},
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
