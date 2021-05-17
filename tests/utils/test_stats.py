import pytest
import numpy as np
import reciprocalspaceship as rs
import gemmi

@pytest.mark.parametrize("cell_and_spacegroup", [
    (gemmi.UnitCell(10., 20., 30., 90., 90., 90.), gemmi.SpaceGroup('P 21 21 21')),
    (gemmi.UnitCell(30., 30., 30., 90., 90., 120.), gemmi.SpaceGroup('R 32')),
])
@pytest.mark.parametrize("anomalous", [False, True])
@pytest.mark.parametrize("full_asu", [False, True])
def test_compute_redundancy(cell_and_spacegroup, anomalous, full_asu):
    """ 
    Test reciprocalspaceship.utils.compute_redundancy.
    """
    dmin = 5.
    cell,spacegroup = cell_and_spacegroup
    hkl = rs.utils.generate_reciprocal_asu(cell, spacegroup, dmin, anomalous=anomalous)
    mult = np.random.choice(10, size=len(hkl))
    hobs = np.repeat(hkl, mult, axis=0)
    hunique, counts = rs.utils.compute_redundancy(hobs, cell, spacegroup, full_asu=full_asu, anomalous=anomalous)
    assert hunique.dtype == np.int32
    assert counts.dtype == np.int32
    assert len(hkl) == len(mult)
    assert len(np.unique(hobs, axis=0)) == np.sum(counts > 0)
    assert np.all(counts[counts>0] == mult[mult > 0])

