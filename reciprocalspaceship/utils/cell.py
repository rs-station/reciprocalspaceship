import numpy as np

from reciprocalspaceship.decorators import cellify


@cellify
def compute_dHKL(H, cell):
    """
    Compute the real space lattice plane spacing, d, associated with
    miller indices and cell.

    Parameters
    ----------
    H : array
        An nx3 array of numerical miller indices.
    cell : tuple, list, np.ndarray of cell parameters, or gemmi.UnitCell
        Unit cell parameters

    Returns
    -------
    dHKL : array
        Array of floating point d spacings in Å.
    """
    # Compress the hkls so we don't do redudant computation
    H = np.array(H, dtype=np.float32)
    hkls, inverse = np.unique(H, axis=0, return_inverse=True)
    F = np.array(cell.fractionalization_matrix.tolist()).astype(np.float64)
    dhkls = np.reciprocal(np.linalg.norm((hkls @ F), 2, 1)).astype(np.float32)
    return dhkls[inverse]


def get_gridsize(dataset, sample_rate=3.0, dmin=None):
    """
    Determine an appropriate 3D grid size for the provided rs.DataSet object
    to represent data with the given `sample_rate` and `dmin`.

    This function will return the smallest grid that yields a real-space grid
    spacing of at most `dmin`/`sample_rate` (in Å) that is 'FFT-friendly',
    meaning that each dimension has 2, 3, or 5 as the largest prime factors.
    The grid will also obey any symmetry relations for the spacegroup of `dataset`.

    Notes
    -----
    - With `dmin=None`, this function sets `dmin` to the highest resolution
      reflection
    - If `dmin` is provided, the returned grid size yields a real-space grid
      spacing that is at most `dmin`/`sample_rate`

    Parameters
    ----------
    dataset : rs.DataSet
        DataSet object for 3D grid
    sample_rate : float
        Sets the minimal grid spacing relative to dmin. For example,
        `sample_rate=3` corresponds to a real-space sampling of dmin/3.
        (default: 3.0)
    dmin : float
        Highest-resolution reflection to consider for grid size. If None,
        `dmin` will be set to the highest resolution reflection in the
        dataset (default: None)

    Returns
    -------
    list(int, int, int)
        Grid size with desired spacing (list of 3 integers)
    """
    dHKL = dataset.compute_dHKL()["dHKL"]
    if dmin is None:
        dmin = dHKL.min()

    # Determine minimal grid size, then compute closest valid grid size
    abc = np.array(dataset.cell.parameters[:3])
    min_spacing = dmin / sample_rate
    min_size = np.ceil(abc / min_spacing).astype(int)
    ds = dataset.loc[dHKL >= dmin]
    return ds.to_gemmi().get_size_for_hkl(min_size=min_size, sample_rate=sample_rate)


@cellify
def generate_reciprocal_cell(cell, dmin, dtype=np.int32):
    """
    Generate the miller indices of the full P1 reciprocal cell.

    Parameters
    ----------
    cell : tuple, list, np.ndarray of cell parameters, or gemmi.UnitCell
        Unit cell parameters
    dmin : float
        Maximum resolution of the data in Å
    dtype : np.dtype (optional)
        The data type of the returned array. The default is np.int32.


    Returns
    -------
    hkl : np.array(int32)
    """
    hmax, kmax, lmax = cell.get_hkl_limits(dmin)
    hkl = np.meshgrid(
        np.linspace(-hmax, hmax + 1, 2 * hmax + 2, dtype=dtype),
        np.linspace(-kmax, kmax + 1, 2 * kmax + 2, dtype=dtype),
        np.linspace(-lmax, lmax + 1, 2 * lmax + 2, dtype=dtype),
    )
    hkl = np.stack(hkl).reshape((3, -1)).T

    # Remove reflection 0,0,0
    hkl = hkl[np.any(hkl != 0, axis=1)]

    # Remove reflections outside of resolution range
    dHKL = cell.calculate_d_array(hkl).astype("float32")
    hkl = hkl[dHKL >= dmin]

    return hkl
