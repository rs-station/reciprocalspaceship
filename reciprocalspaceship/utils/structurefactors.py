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
