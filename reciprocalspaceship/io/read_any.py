from pathlib import Path

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
        filetype = Path(filename).suffix.removeprefix(".")

    match filetype:

        case 'mtz' | 'MTZ':
            return _attempt_read(filename, read_mtz, *args, **kwargs)

        case 'cif' | 'sfcif' | 'CIF' | 'SFCIF':
            return _attempt_read(filename, read_cif, *args, **kwargs)

        case 'pkl' | 'PKL' | 'pickle':
            return _attempt_read(filename, read_pickle, *args, **kwargs)

        case 'stream' | 'crystfel':
            return _attempt_read(filename, read_crystfel, *args, **kwargs)

        case 'ii' | 'II' | 'hkl' | 'HKL' | 'precognition':
            return _attempt_read(filename, read_precognition, *args, **kwargs)

        case 'csv' | 'CSV':
            return _attempt_read(filename, read_csv, *args, **kwargs)

        case 'refl'| 'REFL' | 'dials' | 'Dials':
            return _attempt_read(filename, read_dials_stills, *args, **kwargs)

        case _:
            raise ValueError(f"Unknown file type: {filetype}"
                             "\n            "
                             "Maybe you meant to specify a different filetype? "
                             )


def _attempt_read(filename, read_method, *args, **kwargs):
    try:
        return read_method(filename, *args, **kwargs)
    except Exception as e:
        raise Exception(f"Attempting to read '{filename}' using `{read_method}` raised an exception. " "\n           "
                        "Maybe you meant to specify a different filetype? " "\n           "
                        f"The exception was: {e}")
