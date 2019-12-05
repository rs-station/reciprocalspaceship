import numpy as np
import reciprocalspaceship as rs

def to_structurefactor(sfamps, phases):
    """
    Convert structure factor amplitude and phase to complex structure
    factors

    Parameters
    ----------
    sfamps : CrystalSeries or array-like
        Structure factor amplitudes
    phases : CrystalSeries or array-like
        Phases corresponding to structure factors

    Returns
    -------
    sfs : np.ndarray
        Array of complex-valued structure factors
    """
    if isinstance(sfamps, rs.CrystalSeries):
        sfamps = sfamps.to_numpy()
    if isinstance(phases, rs.CrystalSeries):
        phases = phases.to_numpy()
    return sfamps*np.exp(1j*np.deg2rad(phases))

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
    (sf, phase) : tuple of CrystalSeries
        Tuple of CrystalSeries for the structure factor amplitudes and 
        phases corresponding to the provided complex structure factors
    """
    sf = rs.CrystalSeries(np.abs(sfs), name="F").astype("SFAmplitude")
    phase = rs.CrystalSeries(np.angle(sfs, deg=True), name="Phi").astype("Phase")
    return sf, phase

def compute_internal_differences(crystal, symop, sf_key, err_key=None, phase_key=None):
    """
    Compute internal difference map in provided Crystal object based on
    given symmetry operator.

    Parameters
    ----------
    crystal : Crystal
        Reflections in reduced-symmetry space group
    symop : gemmi.Op
        Space group symmetry operator for internal difference map
    sf_key : str
        Column of structure factor amplitudes
    err_key : str
        Column of errors associated with structure factor amplitudes. 
        Errors are propagated in quadrature.
    phase_key : str
        Column of phases

    Return
    ------
    Crystal
        Crystal object representing internal differences based on given
        symmetry operator
    """

    # Provided symop should not be in reduced symmetry space group
    operators = [ op.triplet() for op in crystal.spacegroup.operations() ]
    if symop.triplet() in operators:
        raise ValueError(f"Given symmetry operation, {symop.triplet()}, "
                         f"is present in space group {mtz.spacegroup.number}")

    columns = [sf_key]
    if err_key:
        columns.append(err_key)
    if phase_key:
        columns.append(phase_key)

    Fplus  = crystal.copy()[columns]
    Fminus = crystal.copy().unmerge_anomalous()[columns]

    # Apply symop to Fminus
    Fminus.apply_symop(symop, inplace=True)
    hkls = Fplus.index
    Fminus = Fminus.loc[Fminus.index.intersection(hkls)]
    Fplus = Fplus.loc[Fminus.index]
    Fminus.sort_index(inplace=True)
    Fplus.sort_index(inplace=True)

    # If phases are given, handle structure factors as complex numbers
    if phase_key:
        sfplus  = to_structurefactor(Fplus[sf_key], Fplus[phase_key])
        sfminus = to_structurefactor(Fminus[sf_key], Fminus[phase_key])
        deltaF, deltaPhi = from_structurefactor(sfplus - sfminus)
        F = Fplus.copy().reset_index()
        F[sf_key] = deltaF
        F[phase_key] = deltaPhi
        F.set_index(["H", "K", "L"], inplace=True)
        F.rename(columns={sf_key:"DF", phase_key:"DPhi"}, inplace=True)
    else:
        deltaF = Fplus[sf_key] - Fminus[sf_key]
        F = Fplus.copy().reset_index()
        F[sf_key] = deltaF.values
        F.set_index(["H", "K", "L"], inplace=True)
        F.rename(columns={sf_key:"DF"}, inplace=True)
        
    # Propagate Errors in quadrature
    if err_key:
        F[err_key] = np.sqrt(Fplus[err_key]**2 + Fminus[err_key]**2)
        F.rename(columns={err_key:"SigDF"}, inplace=True)
        
    return F

    