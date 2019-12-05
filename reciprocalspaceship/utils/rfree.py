import numpy as np
from reciprocalspaceship.dtypes import MTZIntDtype


def add_rfree(crystal, fraction=0.05, bins=20, inplace=False):
    """
    Add an r-free flag to the crystal object for refinement. 
    R-free flags are used to identify reflections which are not used in automated refinement routines.
    This is the crystallographic refinement version of cross validation.

    Parameters
    ----------
    crystal : rs.Crystal
        Crystal object for which to compute a random fraction. 
    fraction : float, optional
        Fraction of reflections to be added to the r-free. (the default is 0.05)
    bins : int, optional
        Number of resolution bins to divide the free reflections over. (the default is 20)
    inplace : bool, optional

    Returns:
    --------
    result : rs.Crystal

    """
    if not inplace:
        crystal = crystal.copy()
    dHKL_present = 'dHKL' in crystal
    if not dHKL_present:
        crystal = crystal._compute_dHKL()

    bin_edges = np.percentile(crystal['dHKL'], np.linspace(100, 0, bins+1))
    bin_edges = np.vstack([bin_edges[:-1], bin_edges[1:]]).T

    crystal['R-free-flags'] = 0
    crystal['R-free-flags'] = crystal['R-free-flags'].astype(MTZIntDtype())

    free = np.random.random(len(crystal)) <= fraction

    for i in range(bins):
        dmax,dmin = bin_edges[i]
        crystal[free & (crystal['dHKL'] >= dmin) & (crystal['dHKL'] <= dmax)] = i

    if not dHKL_present:
        del(crystal['dHKL'])

    return crystal

def copy_rfree(crystal, crystal_with_rfree, inplace=False):
    """
    Copy the rfree flag from one crystal object to another.
    Parameters
    ----------
    crystal : rs.Crystal
        A crystal without an r-free flag or with an undesired one.
    crystal_with_rfree : rs.Crystal
        A crystal with desired r-free flags.
    inplace : bool, optional

    Returns:
    result : rs.Crystal
    """
    if not inplace:
        crystal = crystal.copy()

    crystal['R-free-flags'] =  0
    crystal['R-free-flags'] = crystal['R-free-flags'].astype(MTZIntDtype())
    idx = crystal.index.intersection(crystal_with_rfree.index)
    crystal.loc[idx, "R-free-flags"] = crystal_with_rfree.loc[idx, "R-free-flags"]
    return crystal
