import numpy as np

def compute_dHKL(H, cell):
    """
    Compute the real space lattice plane spacing, d, associated with
    miller indices and cell. 

    Parameters
    ----------
    H : array
        An nx3 array of numerical miller indices. 
    cell : gemmi.UnitCell
        The gemmi UnitCell object with the cell parameters.

    Returns
    -------
    dHKL : array
        Array of floating point d spacings in Å.
    """
    #Compress the hkls so we don't do redudant computation
    H = np.array(H, dtype=np.float32)
    hkls,inverse = np.unique(H, axis=0, return_inverse=True)
    A = np.array(cell.orthogonalization_matrix.tolist()).astype(np.float32)
    dhkls = 1./np.linalg.norm((hkls@np.linalg.inv(A)), 2, 1)
    return dhkls[inverse]
