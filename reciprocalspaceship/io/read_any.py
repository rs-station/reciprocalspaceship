

from reciprocalspaceship.io.crystfel import read_crystfel
from reciprocalspaceship.io.csv import read_csv
from reciprocalspaceship.io.dials import read_dials_stills
from reciprocalspaceship.io.mtz import (
    read_cif,
    read_mtz,
)
from reciprocalspaceship.io.pickle import read_pickle
from reciprocalspaceship.io.precognition import read_precognition


def read(filename,
         filetype='auto',
         *args, **kwargs):
    """
    Populate the dataset object with data from any file format readable by reciprocalspaceship

    If you would only like to read one type of file, use the appropriate read_ function.

    Parameters
    ----------
    filename : str or list or tuple
        name of the file to read (or iterable of files for read_dials_stills)
    filetype : str (optional)
        defaults to 'auto'
    args, kwargs
        Any additional arguments are passed to the file-specific read function.

    Returns
    -------
    DataSet
    """

    if filetype == 'auto':
        filetype = filename.split('.')[-1]

    if filetype in ('mtz', 'MTZ'):
        return _attempt_read(filename, read_mtz, *args, **kwargs)

    if filetype in ('cif', 'sfcif', 'CIF'):
        return _attempt_read(filename, read_cif, *args, **kwargs)

    if filetype in ('pkl', 'PKL', 'pickle'):
        return _attempt_read(filename, read_pickle, *args, **kwargs)

    if filetype in ('crystfel',):
        return _attempt_read(filename, read_crystfel, *args, **kwargs)

    if filetype in ('ii', 'II', 'precognition'):
        return _attempt_read(filename, read_precognition, *args, **kwargs)

    if filetype in ('csv', 'CSV'):
        return _attempt_read(filename, read_csv, *args, **kwargs)


def _attempt_read(filename, read_method, *args, **kwargs):
    try:
        return read_method(filename, *args, **kwargs)
    except Exception as e:
        raise Exception((f"Attempting to read '{filename}' using `{read_method}` raised an exception. " "\n           "
                         "Maybe you meant to specify a different filetype? " "\n           "
                         f"The exception was: {e}"))
