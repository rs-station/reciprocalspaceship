import numpy as np
from gemmi import GroupOps, SpaceGroup

import reciprocalspaceship as rs
from reciprocalspaceship.decorators import spacegroupify


def to_structurefactor(sfamps, phases):
    """
    Convert structure factor amplitude and phase to complex structure
    factors

    Parameters
    ----------
    sfamps : DataSeries or array-like
        Structure factor amplitudes
    phases : DataSeries or array-like
        Phases corresponding to structure factors

    Returns
    -------
    sfs : np.ndarray
        Array of complex-valued structure factors
    """
    if isinstance(sfamps, rs.DataSeries):
        sfamps = sfamps.to_numpy()
    if isinstance(phases, rs.DataSeries):
        phases = phases.to_numpy()
    return sfamps * np.exp(1j * np.deg2rad(phases))


def from_structurefactor(sfs):
    """
    Convert complex structure factors into structure factor amplitudes
    and phases

    Parameters
    ----------
    sfs : np.ndarray
        array of complex structure factors to be converted

    Returns
    -------
    (sf, phase) : tuple of DataSeries
        Tuple of DataSeries for the structure factor amplitudes and
        phases corresponding to the provided complex structure factors
    """
    index = None
    if isinstance(sfs, rs.DataSeries):
        index = sfs.index
    sf = rs.DataSeries(np.abs(sfs), index=index, name="F").astype("SFAmplitude")
    phase = rs.DataSeries(np.angle(sfs, deg=True), index=index, name="Phi").astype(
        "Phase"
    )
    return sf, phase


@spacegroupify
def compute_structurefactor_multiplicity(H, sg, include_centering=True):
    """
    Compute the multiplicity of each reflection in ``H``.

    Parameters
    ----------
    H : array
        n x 3 array of Miller indices
    spacegroup : str, int, gemmi.SpaceGroup
        The space group to identify the asymmetric unit
    include_centering : bool
        Whether or not to include the multiplicity inherent in the lattice centering.
        The default is True. If False, the minimum value of epsilon will be 1
        irrespective of space group.

    Returns
    -------
    epsilon : array
        an array of length n containing the multiplicity
        of each hkl.
    """
    group_ops = sg.operations()
    is_centric = group_ops.is_centric()

    # Lookup based on centering is equivalent to counting the number of translational
    # centering operations. Using the number of operations has proven more robust.
    if include_centering:
        L = 1 + is_centric
    else:
        L = len(group_ops.cen_ops)
        L = L * (1 + is_centric)

    eps = np.zeros(len(H))
    for op in group_ops:
        h = rs.utils.apply_to_hkl(H, op)
        if is_centric:
            eps += np.all(h == H, 1) | np.all(h == -H, 1)
        else:
            eps += np.all(h == H, 1)
    return eps / L


@spacegroupify
def is_centric(H, spacegroup):
    """
    Determine if Miller indices are centric in a given spacegroup

    Parameters
    ----------
    H : array
        n x 3 array of Miller indices
    spacegroup : str, int, gemmi.SpaceGroup
        The space group in which to classify centrics

    Returns
    -------
    centric : array
        Boolean arreay with len(centric) == np.shape(H)[0] == n
    """
    group_ops = spacegroup.operations()
    hkl, inverse = np.unique(H, axis=0, return_inverse=True)
    centric = group_ops.centric_flag_array(hkl)
    return centric[inverse]


@spacegroupify
def is_absent(H, spacegroup):
    """
    Determine if Miller indices are systematically absent in a given
    spacegroup.

    Parameters
    ----------
    H : array
        n x 3 array of Miller indices
    spacegroup : str, int, gemmi.SpaceGroup
        The space group in which to classify systematic absences

    Returns
    -------
    absent : array
        Boolean array of length n. absent[i] == True if H[i] is systematically absent in sg.
    """
    return spacegroup.operations().systematic_absences(H)
