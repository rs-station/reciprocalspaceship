import pandas as pd
from reciprocalspaceship import DataSeries

def read_pickle(filepath_or_buffer):
    """
    Load pickled reciprocalspaceship object from file.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        Pickled object to be read

    Returns
    -------
    unpickled : same type as object stored in file

    See Also
    --------
    DataSet.to_pickle : Pickle object to file
    """
    unpickled = pd.read_pickle(filepath_or_buffer)

    # Ensure DataSeries objects are not promoted to DataSet
    if isinstance(unpickled, DataSeries):
        return unpickled

    # Clean up datatypes for index
    index_labels = unpickled.index.names
    unpickled = unpickled.reset_index()
    unpickled = unpickled.set_index(index_labels)

    return unpickled
