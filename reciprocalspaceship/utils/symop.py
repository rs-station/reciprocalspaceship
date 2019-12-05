import numpy as np

def apply_to_hkl(H, op):
    """
    Apply symmetry operator to hkls.

    Parameters
    ----------
    H : array
        n x 3 array of Miller indices
    op : gemmi.Op
        gemmi symmetry operator to be applied

    Returns
    -------
    result : array
        n x 3 array of Miller indices after operator application
    """
    return np.floor_divide(np.matmul(H, op.rot), op.DEN)

def phase_shift(H, op):
    """
    Calculate phase shift for symmetry operator. 

    Parameters
    ----------
    H : array
        n x 3 array of Miller indices
    op : gemmi.Op
        gemmi symmetry operator to be applied

    Returns
    -------
    result : array
        array of phase shifts
    """
    return -2*np.pi*np.matmul(H, op.tran) / op.DEN
