import logging
import warnings
from contextlib import contextmanager
from importlib.util import find_spec


def check_for_mpi():
    try:
        from mpi4py import MPI

        return True
    except Exception as err:
        message = (
            f"Failed `from mpi4py import MPI` with {err}. Falling back to serial mode."
        )
        warnings.warn(message, ImportWarning)
        return False
