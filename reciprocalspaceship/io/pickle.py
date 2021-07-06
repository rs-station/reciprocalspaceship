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
    dataset = pd.read_pickle(filepath_or_buffer)

    # Clean up datatypes for index
    index_labels = dataset.index.names
    dataset.reset_index(inplace=True)
    dataset.set_index(index_labels, inplace=True)

    return dataset
