import numpy as np

def bin_by_percentile(data, bins=20, ascending=True, format_str=".2f"):
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
    if ascending:
        order = 1
        right = False
    else:
        order = -1
        right = True
        
    bin_edges = np.percentile(data, np.linspace(0, 100, bins+1)[::order])

    # 0-index bins and ensure that right-most datapoint is in correct bin
    assignments = np.digitize(data, bins=bin_edges, right=right) - 1
    rightmost = np.where(assignments == bins)[0]
    assignments[rightmost] -= 1

    bin_labels = [ f"{edge1:{format_str}} - {edge2:{format_str}}" for edge1, edge2 in zip(bin_edges[0:-1], bin_edges[1:]) ]

    return assignments, bin_labels
