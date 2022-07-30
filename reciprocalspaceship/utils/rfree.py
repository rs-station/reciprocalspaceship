import numpy as np

from reciprocalspaceship.dtypes import MTZIntDtype


def add_rfree(dataset, fraction=0.05, ccp4_convention=False, inplace=False, seed=None):
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
    ccp4_conventiion: bool, optional
        Whether to use the CCP4 convention for R-free flags (Default: False).
        If True, use CCP4 convention:
        0 is test set, 1 is working set, and the key is "FreeR_flag".
        If False, use Phenix(CNS/XPOLAR) convention:
        1 is test set, 0 is working set, and key is "R-free-flags".
        See https://www.ccp4.ac.uk/html/freerflag.html#description for convention details.
    inplace : bool, optional
    seed : int, optional
        Seed to be passed to numpy.random.default_rng random number generator
        for reproducible r-free flags. If None (default), r-free flags will
        different each time.

    Returns
    -------
    result : rs.DataSet

    """
    if not inplace:
        dataset = dataset.copy()

    rng = np.random.default_rng(seed)
    test_set = rng.random(len(dataset)) <= fraction

    if not ccp4_convention:
        rfree_key = "R-free-flags"
        update = test_set
    else:
        rfree_key = "FreeR_flag"
        update = ~test_set

    dataset[rfree_key] = 0
    dataset[rfree_key] = dataset[rfree_key].astype(MTZIntDtype())
    dataset.loc[update, rfree_key] = 1
    return dataset


def copy_rfree(dataset, dataset_with_rfree, inplace=False, rfree_key=None):
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
    rfree_key : str, optional
        Name of the column containing rfree flags in dataset_with_rfree.
        If None, dataset_with_rfree will be checked for column "R-free-flags"
        (phenix convention) then column "FreeR_flag" (ccp4 convention)

    Returns
    -------
    result : rs.DataSet
    """
    if not inplace:
        dataset = dataset.copy()

    if rfree_key is not None:
        if rfree_key not in dataset_with_rfree.columns:
            raise ValueError(
                f"""Supplied dataset_with_rfree contains no column {rfree_key}"""
            )
    elif "R-free-flags" in dataset_with_rfree.columns:
        rfree_key = "R-free-flags"
    elif "FreeR_flag" in dataset_with_rfree.columns:
        rfree_key = "FreeR_flag"
    else:
        raise ValueError(
            """Failed to automatically find r-free flags in dataset_with_rfree. Please supply an rfree_key"""
        )

    dataset[rfree_key] = 0
    dataset[rfree_key] = dataset[rfree_key].astype(MTZIntDtype())
    idx = dataset.index.intersection(dataset_with_rfree.index)
    dataset.loc[idx, rfree_key] = dataset_with_rfree.loc[idx, rfree_key]
    return dataset
