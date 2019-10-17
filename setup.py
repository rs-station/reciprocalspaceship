from setuptools import setup, find_packages

setup(
    name='reciprocalspaceship',
    version='0.1.1',
    author='Kevin M. Dalton; Jack B. Greisman',
    author_email='kmdalton@g.harvard.edu; greisman@g.harvard.edu',
    packages=find_packages(),
    description='Tools for exploring reciprocal space',
    install_requires=[
        "gemmi >= 0.3.1"
    ],
)
