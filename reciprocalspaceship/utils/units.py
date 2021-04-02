from scipy.constants import Planck, c, electron_volt


def ev2angstroms(ev):
    return Planck * c / ev / 1e-10 / electron_volt

def angstroms2ev(angstroms):
    return Planck * c / angstroms / 1e-10 / electron_volt


