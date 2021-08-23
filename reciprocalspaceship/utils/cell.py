import reciprocalspaceship as rs
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


def generate_reciprocal_cell(cell, dmin, dtype=np.int32):
    """
    Generate the miller indices of the full P1 reciprocal cell. 

    Parameters
    ----------
    cell : gemmi.UnitCell
        A gemmi cell object.
    dmin : float
        Maximum resolution of the data in Å
    dtype : np.dtype (optional)
        The data type of the returned array. The default is np.int32.
        

    Returns
    -------
    hkl : np.array(int32)
    """
    hmax,kmax,lmax = cell.get_hkl_limits(dmin)
    hkl = np.meshgrid(
        np.linspace(-hmax-2, hmax+3, 2*hmax+6, dtype=dtype),
        np.linspace(-kmax-2, kmax+3, 2*kmax+6, dtype=dtype),
        np.linspace(-lmax-2, lmax+3, 2*lmax+6, dtype=dtype),
    )
    hkl = np.stack(hkl).reshape((3, -1)).T
    #Remove reflection 0,0,0
    hkl = hkl[np.any(hkl != 0, axis=1)]
    #Remove reflections outside of resolution range
    hkl = hkl[cell.calculate_d_array(hkl) >= dmin]
    return hkl

