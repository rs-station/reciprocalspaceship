.. _style:

Code Style
==========

This library tries to maintain a consistent code style in order to make it
easier for users to inspect our code and for us to review new contributions.
We try to follow the `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ standard,
and use `Black <https://black.readthedocs.io/en/stable/>`_ to establish consistent
formatting throughout our library.

Pre-commit
----------

In order to make it easy to follow our code style, we provide
`pre-commit hooks <https://pre-commit.com/>`_ to automatically run our style checks and
to fix issues. Although it is possible to run each of our tests individually, this provides
a mechanism to automatically run our suite of style checks at each commit.

In order to set up our pre-commit hooks, first install ``pre-commit``::

  pip install pre-commit

or::

  conda install -c conda-forge pre-commit

To install our pre-commit hooks to run at every ``git commit``, run the following from the
home directory of the git repo::

  pre-commit install

If you don't want to use ``pre-commit`` as an automatic part of you workflow, you can always
run the commands and checks  with the following::

  pre-commit run --files <files to be autoformatted>

Pre-commit and Pull Requests
----------------------------

Regardless of whether you opt to run our ``pre-commit hooks`` locally, they will automatically be
run on any pull requests. This automation is handled by `pre-commit ci <https://pre-commit.ci/>`_,
and is intended to make it as easy as possible to follow our style guidelines.
