import numpy as np


def assign_with_binedges(data, bin_edges, right_inclusive=True):
    """
    Assign data using given bin edges. This function assumes that the bin edges
    are monotonic, and that the resulting bins contain every entry in `data`.

    If the smallest and largest values in `data` exactly match a bin edge,
    they are assigned to the relevant bin, regardless of the value of `right`.

    Parameters
    ----------
    data : list, np.ndarray or Series-like
        Data to assign to bins
    bin_edges : list or np.ndarray
        Edge values for each bin. Must be monotonically ascending or descending
    right_incusive : bool
        Whether the right edge of each bin should be considered inclusive or
        exclusive
    """
    # Check bin_edges are strictly ascending or descending
    if not (
        np.all(bin_edges[1:] >= bin_edges[:-1])
        or np.all(bin_edges[1:] <= bin_edges[:-1])
    ):
        raise ValueError(
            f"Given `bin_edges` are not strictly ascending or descending: {bin_edges}"
        )

    # Check bin_edges contain every data point
    if (min(data) < min(bin_edges)) or (max(data) > max(bin_edges)):
        raise ValueError(
            "This function assumes `bin_edges` contain every entry in `data`"
        )

    ascending = bin_edges[0] < bin_edges[-1]
    assignments = np.digitize(data, bins=bin_edges, right=right_inclusive) - 1

    # Fix biggest or smallest entry
    if right_inclusive:
        if ascending:
            smallest = np.where(data == bin_edges[0])[0]
            assignments[smallest] += 1
        else:
            smallest = np.where(data == bin_edges[-1])[0]
            assignments[smallest] -= 1

    else:
        if ascending:
            biggest = np.where(data == bin_edges[-1])[0]
            assignments[biggest] -= 1
        else:
            biggest = np.where(data == bin_edges[0])[0]
            assignments[biggest] += 1

    return assignments


def bin_by_percentile(
    data, bins=20, ascending=True, return_edges=False, format_str=".2f"
):
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
    return_edges: bool
        Whether to return the bin edges
    format_str : str
        Format string for constructing bin labels

    Return
    ------
    assignments : np.ndarray
        Bins to which data were assigned
    bin_labels : list
        Labels denoting bin edges
    bin_edges : np.ndarray (optional)
        If `return_edges=True`, an array with the bin edges is returned
    """
    if ascending:
        order = 1
        right = False
    else:
        order = -1
        right = True

    bin_edges = np.percentile(data, np.linspace(0, 100, bins + 1)[::order])

    assignments = assign_with_binedges(data, bin_edges, right_inclusive=right)

    bin_labels = [
        f"{edge1:{format_str}} - {edge2:{format_str}}"
        for edge1, edge2 in zip(bin_edges[0:-1], bin_edges[1:])
    ]

    if return_edges:
        return assignments, bin_labels, bin_edges
    else:
        return assignments, bin_labels
