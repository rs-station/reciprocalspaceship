import pandas as pd

def read_pickle(filepath_or_buffer):
    """
    Load pickled DataSet object from file.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        Pickled object to be read

    Returns
    -------
    unpickled : same type as object stored in file
    """
    return pd.read_pickle(filepath_or_buffer)
