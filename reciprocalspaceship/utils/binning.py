import numpy as np

def bin_by_percentile(data, bins=20, ascending=True, format_str=".3f"):
    """
    Bin data by percentile.

    Parameters
    ----------
    data : list, np.ndarray or Series-like
        Data to bin by percentile
    bins : int
        Number of bins
    ascending : bool
        Whether to bin data by value from low to high
    format_str : str
        Format string for constructing bin labels

    Return
    ------
    assignments, bin_labels
        Bins to which data were assigned, and corresponding labels 
        denoting value ranges
    """
    bin_edges = np.percentile(data, np.linspace(0, 100, bins+1)[::order])
    assignments = np.digitize(data, bins=bin_edges)
    bin_labels = np.array([ f"{edge1:{format_str}} - {edge2:{format_str}}" for edge1, edge2 in zip(bin_edges[0:-1], bin_edges[1:]) ])
    return assignments, bin_labels
