.. _documentation:

Building Documentation
======================

The ``reciprocalspaceship`` documentation is built using `sphinx <http://sphinx-doc.org/>`_. The documentation is automatically rebuilt upon any ``git push`` action to
the ``main`` branch of the repo; however, it is also possible to build it locally in order to preview any changes.

Dependencies
------------

There are a few additional dependencies that are needed to build the documentation. These can be installed by running the following command in your cloned (or forked) repository::

  pip install -e .[dev]

Local Preview of Documentation
------------------------------
  
The documentation can then be built using the ``Makefile`` in the ``docs/`` subdirectory::

  cd docs
  make html

The ``index.html`` of the documentation will then be found in ``docs/_build/``, and can be viewed in any browser. 

If you have a partial build of the documentation, or if you make a change to the reStructuredText (RST) templates,
it may be necessary to remove the existing documentation files. This can be accomplished by running the following
command in the ``docs/`` subdirectory::

  make clean

Pushing Changes to Documentation
--------------------------------

The documentation is rebuilt by default upon any ``git push`` action to the master branch. This occurs when a Pull
Request is merged into the master branch. As such, please feel free to propose changes to the documentation via a
Pull Request. The changes will take place when the Pull Request is merged into the project repository.

