.. _contributing:

Contributing to ``rs``
=======================================

We encourage you to file `Issues on GitHub <https://github.com/rs-station/reciprocalspaceship/issues>`_ if you run into any problems when using
``reciprocalspaceship``. This is also a great place to request features, suggest improvements, or
ask any questions.

We also invite you to file Pull Requests in order to help us maintain this library.
No Pull Request is too small (in fact, we encourage you to only tackle single tasks in a PR).
Contributions to the library, tests, and documentation are all welcome.

Setting up a Dev Environment
----------------------------

There are additional dependencies that are needed for running the testing suite, building documentation, and enforcing our code style.
These can be installed by running the following commands from the project home directory::

  pip install -e .[dev]
  pip install pre-commit
  pre-commit install

The above commands install a few extra dependencies, and also add a pre-commit hook that is used to enforce our code style. There are more
detailed sections in this documentation that cover our `code style <style.rst>`_, `testing framework <testing.rst>`_, and `documentation <documentation.rst>`_.
