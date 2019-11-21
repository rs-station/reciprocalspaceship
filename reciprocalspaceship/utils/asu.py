import numpy as np

#The 11 laue group symbols as per gemmi
laue_groups = {'-1',
 '-3',
 '-3m',
 '2/m',
 '4/m',
 '4/mmm',
 '6/m',
 '6/mmm',
 'm-3',
 'm-3m',
 'mmm'
}

def in_asu(H, SpaceGroup):
    """
    Check to see if Miller indices are in the asymmetric unit of a space group.

    Parameters
    ----------
    H : array
        n x 3 array of Miller indices
    SpaceGroup : gemmi.SpaceGroup
        The space group to identify the asymmetric unit
    
    Returns
    -------
    in_asu : array
        Array of bools with length n. 
    """

    h,k,l = H.T

    
