from functools import wraps
from inspect import signature


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
