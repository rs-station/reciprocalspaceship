import numpy as np

import reciprocalspaceship as rs


def test_ev2angstroms():
    angstroms = np.linspace(0.1, 100, 1000)
    ev = rs.utils.angstroms2ev(angstroms)
    assert np.allclose(ev * angstroms, 12398.41984332)


def test_angstroms2ev():
    ev = np.linspace(1000.0, 50000.0, 1000)
    angstroms = rs.utils.angstroms2ev(ev)
    assert np.allclose(ev * angstroms, 12398.41984332)
