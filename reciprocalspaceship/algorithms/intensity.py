import numpy as np

import reciprocalspaceship as rs


def compute_intensity_from_structurefactor(F, SigF):
    """
    Back-calculate intensities and approximate intensity error estimates from
    structure factor amplitudes and error estimates

    Intensity computed as I = SigF*SigF + F*F. Intensity error estimate
    approximated as SigI = abs(2*F*SigF)

    Parameters
    ----------
    F : DataSeries or array-like
        Structure factor amplitudes
    SigF : DataSeries or array-like
        Phases corresponding to structure factors

    Returns
    -------
    (I, sigI) : tuple of np.ndarray
        Tuple of np.ndarray containing back-calculated intensities and
        approximate intensity error estimates
    """

    if isinstance(F, rs.DataSeries):
        F = F.to_numpy()
    elif isinstance(F, list):
        F = np.array(F)

    if isinstance(SigF, rs.DataSeries):
        SigF = SigF.to_numpy()
    elif isinstance(SigF, list):
        SigF = np.array(SigF)

    I = SigF * SigF + F * F
    SigI = np.abs(2 * F * SigF)
    return (I, SigI)
