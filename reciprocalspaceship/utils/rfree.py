import numpy as np
from reciprocalspaceship.dtypes import MTZIntDtype


def add_rfree(dataset, fraction=0.05, bins=20, inplace=False):
    """
    Add an r-free flag to the dataset object for refinement. 
    R-free flags are used to identify reflections which are not used in automated refinement routines.
    This is the crystallographic refinement version of cross validation.

    Parameters
    ----------
    dataset : rs.DataSet
        Dataset object for which to compute a random fraction. 
    fraction : float, optional
        Fraction of reflections to be added to the r-free. (the default is 0.05)
    bins : int, optional
        Number of resolution bins to divide the free reflections over. (the default is 20)
    inplace : bool, optional

    Returns
    -------
    result : rs.DataSet

    """
    if not inplace:
        dataset = dataset.copy()
    dHKL_present = 'dHKL' in dataset
    if not dHKL_present:
        dataset = dataset.compute_dHKL(inplace=True)

    bin_edges = np.percentile(dataset['dHKL'], np.linspace(100, 0, bins+1))
    bin_edges = np.vstack([bin_edges[:-1], bin_edges[1:]]).T

    dataset['R-free-flags'] = 0
    dataset['R-free-flags'] = dataset['R-free-flags'].astype(MTZIntDtype())

    free = np.random.random(len(dataset)) <= fraction

    for i in range(bins):
        dmax,dmin = bin_edges[i]
        dataset[free & (dataset['dHKL'] >= dmin) & (dataset['dHKL'] <= dmax)] = i

    if not dHKL_present:
        del(dataset['dHKL'])

    return dataset

def copy_rfree(dataset, dataset_with_rfree, inplace=False):
    """
    Copy the rfree flag from one dataset object to another.

    Parameters
    ----------
    dataset : rs.DataSet
        A dataset without an r-free flag or with an undesired one.
    dataset_with_rfree : rs.DataSet
        A dataset with desired r-free flags.
    inplace : bool, optional
        Whether to operate in place or return a copy

    Returns
    -------
    result : rs.DataSet
    """
    if not inplace:
        dataset = dataset.copy()

    dataset['R-free-flags'] =  0
    dataset['R-free-flags'] = dataset['R-free-flags'].astype(MTZIntDtype())
    idx = dataset.index.intersection(dataset_with_rfree.index)
    dataset.loc[idx, "R-free-flags"] = dataset_with_rfree.loc[idx, "R-free-flags"]
    return dataset
