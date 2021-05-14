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
    F = np.array(cell.fractionalization_matrix.tolist()).astype(np.float64)
    dhkls = np.reciprocal(np.linalg.norm((hkls@F), 2, 1)).astype(np.float32)
    return dhkls[inverse]


def generate_reciprocal_cell(cell, dmin):
    """
    Generate the miller indices of the full P1 reciprocal cell. 

    Parameters
    ----------
    cell : gemmi.UnitCell
        A gemmi cell object.
    dmin : float
        Maximum resolution of the data in Å

    Returns
    -------
    hkl : np.array(int)
    """
    hmax,lmax,kmax = cell.get_hkl_limits(dmin)
    hkl = np.mgrid[
        -hmax-1:hmax+2:1,
        -kmax-1:kmax+2:1,
        -lmax-1:lmax+2:1,
    ].reshape((3, -1)).T
    #Remove reflection 0,0,0
    hkl = hkl[np.any(hkl != 0, axis=1)]
    #Remove reflections outside of resolution range
    hkl = hkl[cell.calculate_d_array(hkl) >= dmin]
    return hkl

