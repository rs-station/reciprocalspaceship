import numpy as np

ccp4_hkl_asu = [
  0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,  
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 7, 6, 7, 6, 7, 7, 7,  
  6, 7, 6, 7, 7, 6, 6, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,  
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9,  
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9               
]                                                                              

asu_cases = {
    0 : lambda h,k,l : (l>0) | ((l==0) & ((h>0) | ((h==0) & (k>=0)))),
    1 : lambda h,k,l : (k>=0) & ((l>0) | ((l==0) & (h>=0))),
    2 : lambda h,k,l : (h>=0) & (k>=0) & (l>=0),
    3 : lambda h,k,l : (l>=0) & (((h>=0) & (k>0)) | ((h==0) & (k==0))),
    4 : lambda h,k,l : (h>=k) & (k>=0) & (l>=0),
    5 : lambda h,k,l : ((h>=0) & (k>0)) | ((h==0) & (k==0) & (l>=0)),
    6 : lambda h,k,l : (h>=k) & (k>=0) & ((k>0) | (l>=0)),
    7 : lambda h,k,l : (h>=k) & (k>=0) & ((h>k) | (l>=0)),
    8 : lambda h,k,l : (h>=0) & (((l>=h) & (k>h)) | ((l==h) & (k==h))),
    9 : lambda h,k,l : (k>=l) & (l>=h) & (h>=0),
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
    result : array
        Array of bools with length n. 
    """

    h,k,l = H.T
    H_ref = np.zeros(H.shape, dtype=H.dtype)
    basis_op = SpaceGroup.basisop.inverse()
    for i,h in enumerate(H):
        H_ref[i] = basis_op.apply_to_hkl(h)
    idx = ccp4_hkl_asu[SpaceGroup.number-1] 
    return asu_cases[idx](*H_ref.T)

