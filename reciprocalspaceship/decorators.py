from functools import wraps
from inspect import signature

import gemmi
import numpy as np


def inplace(f):
    """
    A decorator that applies the inplace argument.

    Base function must have a bool param called "inplace" in  the call
    signature. The position of `inplace` argument doesn't matter.
    """

    @wraps(f)
    def wrapped(ds, *args, **kwargs):
        sig = signature(f)
        bargs = sig.bind(ds, *args, **kwargs)
        bargs.apply_defaults()
        if "inplace" in bargs.arguments:
            if bargs.arguments["inplace"]:
                return f(ds, *args, **kwargs)
            else:
                return f(ds.copy(), *args, **kwargs)
        else:
            raise KeyError(
                f'"inplace" not found in local variables of @inplacemethod decorated function {f} '
                "Edit your method definition to include `inplace=Bool`. "
            )

    return wrapped


def range_indexed(f):
    """
    A decorator that presents the calling dataset with a range index.

    This decorator facilitates writing methods that are agnostic to the
    true indices in a DataSet. Original index columns are preserved through
    wrapped function calls.
    """

    @wraps(f)
    def wrapped(ds, *args, **kwargs):
        names = ds.index.names
        ds = ds._index_from_names([None], inplace=True)
        result = f(ds, *args, **kwargs)
        result = result._index_from_names(names, inplace=True)
        ds = ds._index_from_names(names, inplace=True)
        return result.__finalize__(ds)

    return wrapped


def _convert_spacegroup(val):
    """Helper method to try to convert value to gemmi.SpaceGroup"""
    # GH#18: Type-checking for supported input types
    if isinstance(val, gemmi.SpaceGroup) or (val is None):
        return val
    elif isinstance(val, (str, int)):
        return gemmi.SpaceGroup(val)
    else:
        raise ValueError(f"Cannot construct gemmi.SpaceGroup from value: {val}")


def spacegroupify(func=None, *sg_args):
    """
    A decorator that converts spacegroup arguments to gemmi.SpaceGroup objects.

    This decorator facilitates writing methods that use gemmi.SpaceGroup objects
    by allowing the method to accept spacegroup arguments as str, int, or
    gemmi.SpaceGroup without needing to write boilerplate argument checking code.

    When specified as ``@spacegroupify`` or ``@spacegroupify()``, any arguments
    named "spacegroup", "space_group", or "sg" are automatically coerced to provide
    gemmi.SpaceGroup values.

    When specified with argument names, such as ``@spacegroupify("parent_sg")``, only
    the provided argument names are coerced to gemmi.SpaceGroup. The provided argument
    needs to exactly match an argument in the decorated function's call signature
    for this decorator to coerce the input values.

    Note
    ----
        ``None`` values are passed to the decorated function unchanged.
    """
    if not callable(func) and func is not None:
        sg_args = (func, *sg_args)

    if len(sg_args) == 0:
        sg_args = ("spacegroup", "space_group", "sg")

    def _decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            sig = signature(f)
            bargs = sig.bind(*args, **kwargs)
            bargs.apply_defaults()
            for arg in sg_args:
                if arg in bargs.arguments:
                    bargs.arguments[arg] = _convert_spacegroup(bargs.arguments[arg])
            return f(**bargs.arguments)

        return wrapped

    return _decorator(func) if callable(func) else _decorator


def _convert_unitcell(val):
    """Helper method to try to convert value to gemmi.UnitCell"""
    # GH#18: Type-checking for supported input types
    if isinstance(val, gemmi.UnitCell) or (val is None):
        return val
    elif isinstance(val, (list, tuple, np.ndarray)) and len(val) == 6:
        return gemmi.UnitCell(*val)
    else:
        raise ValueError(f"Cannot construct gemmi.UnitCell from value: {val}")


def cellify(func=None, *cell_args):
    """
    A decorator that converts unit cell arguments to gemmi.UnitCell objects.

    This decorator facilitates writing methods that use gemmi.UnitCell objects
    by allowing the method to accept unit cell arguments as tuples, lists, numpy
    arrays, or gemmi.UnitCell without needing to write boilerplate argument
    checking code.

    When specified as ``@cellify`` or ``@cellify()``, any arguments named
    "cell", "unit_cell", or "unitcell" are automatically coerced to provide
    gemmi.UnitCell values.

    When specified with argument names, such as ``@cellify("uc")``, only
    the provided argument names are coerced to gemmi.UnitCell. The provided argument
    needs to exactly match an argument in the decorated function's call signature
    for this decorator to coerce the input values.

    Note
    ----
        ``None`` values are passed to the decorated function unchanged.
    """
    if not callable(func) and func is not None:
        cell_args = (func, *cell_args)

    if len(cell_args) == 0:
        cell_args = ("cell", "unit_cell", "unitcell")

    def _decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            sig = signature(f)
            bargs = sig.bind(*args, **kwargs)
            bargs.apply_defaults()
            for arg in cell_args:
                if arg in bargs.arguments:
                    bargs.arguments[arg] = _convert_unitcell(bargs.arguments[arg])
            return f(**bargs.arguments)

        return wrapped

    return _decorator(func) if callable(func) else _decorator
