import numpy as np
from scipy.constants import Planck, c, electron_volt

_conversion_factor = Planck * c / 1e-10 / electron_volt


def eV2Angstroms(ev):
    out = np.empty_like(ev)
    np.divide(_conversion_factor, ev, out=out)
    return out


def Angstroms2eV(angstroms):
    return eV2Angstroms(angstroms)


# Add legacy aliases
ev2angstroms = eV2Angstroms
angstroms2ev = Angstroms2eV
