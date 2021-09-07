.. _testing:

Running Tests
=============

Crystallography is full of corner cases and gotchas. We strive to make ``reciprocalspaceship``
a useful library for any and all crystallographers, so we try to do our best to write a general-pupose
library that supports all expected use cases. To accomplish this, we are committed to a test-driven
development process.

When contributing to ``rs``, we encourage you to write tests that define the use and expected behavior
of your new feature. The code you contribute should then be written in order to pass your tests.

``pytest`` Testing Framework
----------------------------

Most of the ``rs`` test suite is written in a functional style using the ``pytest`` framework. This
allows us to write functions that test desired features and use assertions to define expected behavior.
``pytest`` also provides useful features for simplifying the testing of different use cases by using
``parametrize`` to test multiple use cases, and by using ``fixture`` to construct common test objects
on a per-test basis. For details about using ``pytest`` please check out their
`documentaiton <https://docs.pytest.org/en/stable/>`_.

Running Individual Tests
------------------------

The full test suite takes a few minutes to run, so it is often useful to only run a few of the tests
during the initial testing phase. This can be easily handled by ``pytest``. It is first necessary to
have pytest installed to your Python environment. Then, it is possible to specify the test to the
files to be run, or a directory in which to find test files. This will run the tests in
``tests/testdir/test_feature.py``::

  python -m pytest -c /dev/null tests/testdir/test_feature.py

This call will run all of the ``test*.py`` files in the ``tests/testdir/`` directory::

  python -m pytest -c /dev/null tests/testdir/

Running the ``rs`` Test Suite
-----------------------------

There are two ways to run the full ``reciprocalspaceship`` test suite. Using the above call, it is
possible to directly call ``pytest`` giving it the full ``tests/`` directory::

  python -m pytest -c /dev/null tests

Alternatively, in the project home directory one can use the ``setup.py`` to run the test suite. This
has the added benefit that any dependencies that are required for testing will automatically be installed.
In addition, ``pytest-xdist`` will be used which allows for one to parallelize testing over all available
cores::

  python setup.py test

Continuous Integration
----------------------

The test suite for ``reciprocalspaceship`` is set up to run on any pull request. It will be necessary to
fix any issues before the changes can be merged into the project repository.
