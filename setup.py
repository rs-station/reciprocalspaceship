from setuptools import setup, find_packages

# Get version number
def getVersionNumber():
    with open("VERSION", 'r') as vfile:
        version = vfile.read().strip()
    return version
__version__ = getVersionNumber()


setup(
    name='reciprocalspaceship',
    version=__version__,
    author='Kevin M. Dalton; Jack B. Greisman',
    author_email='kmdalton@g.harvard.edu; greisman@g.harvard.edu',
    packages=find_packages(),
    description='Tools for exploring reciprocal space',
    install_requires=[
        "gemmi >= 0.3.7",
        "pandas > 1.0"
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],
)
