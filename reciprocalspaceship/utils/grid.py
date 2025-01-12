import gemmi
import numpy as np

from reciprocalspaceship.decorators import cellify, spacegroupify


@cellify
@spacegroupify
def get_reciprocal_grid_size(cell, dmin, sample_rate=3.0, spacegroup=None):
    """
    Determine an appropriate 3D grid size for the provided cell and spacegroup
    to represent data with the given `sample_rate` and `dmin`.

    This function will return the smallest grid that yields a real-space grid
    spacing of at most `dmin`/`sample_rate` (in Å) that is 'FFT-friendly',
    meaning that each dimension has 2, 3, or 5 as the largest prime factors.
    The grid will also obey any symmetry relations for the spacegroup.

    Parameters
    ----------
    cell : tuple, list, np.ndarray of cell parameters, or gemmi.UnitCell
        Unit cell parameters
    dmin : float
        Highest-resolution reflection to consider for grid size
    sample_rate : float
        Sets the minimal grid spacing relative to dmin. For example,
        `sample_rate=3` corresponds to a real-space sampling of dmin/3.
        Value must be >= 1.0 (default: 3.0)
    spacegroup : int, str, gemmi.SpaceGroup, or None
        Spacegroup for imposing symmetry constraints on grid dimensions.
        (default: None)

    Returns
    -------
    list(int, int, int)
        Grid size with desired spacing (list of 3 integers)

    Raises
    ------
    ValueError
        If sample_rate < 1.0
    """
    if sample_rate < 1.0:
        raise ValueError(f"sample_rate must be >= 1.0. Given: {sample_rate}")

    # Determine minimal grid size, then compute closest valid grid size
    abc = np.array(cell.parameters[:3])
    min_spacing = dmin / sample_rate
    min_size = np.ceil(abc / min_spacing).astype(int)

    # Use gemmi.Mtz to find valid grid (FFT-friendly and obeys symmetry)
    m = gemmi.Mtz()
    if spacegroup is not None:
        m.spacegroup = spacegroup

    return m.get_size_for_hkl(min_size=min_size)
