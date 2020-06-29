import pytest
import numpy as np
import reciprocalspaceship as rs

@pytest.mark.parametrize("inplace", [True, False])
def test_hkl_to_asu(mtz_by_spacegroup, inplace):
    """Test DataSet.hkl_to_asu() for common spacegroups"""
    x = rs.read_mtz(mtz_by_spacegroup)
    y = rs.read_mtz(mtz_by_spacegroup[:-4] + '_p1.mtz')
    y.spacegroup = x.spacegroup

    yasu = y.hkl_to_asu(inplace=inplace)
    assert len(x.index.difference(yasu.index)) == 0
    assert len(yasu.index.difference(x.index)) == 0

    Fx    = x.loc[yasu.index, 'FMODEL'].values.astype(float) 
    Fyasu = yasu['FMODEL'].values.astype(float) 
    assert np.allclose(Fx, Fyasu)

    Phx    = x.loc[yasu.index, 'PHIFMODEL'].values.astype(float) 
    Phyasu = yasu['PHIFMODEL'].values.astype(float) 
    Sx    = Fx*np.exp(1j*np.deg2rad(Phx))
    Syasu = Fyasu*np.exp(1j*np.deg2rad(Phyasu))
    assert np.allclose(Sx, Syasu, rtol=1e-3)

    if inplace:
        assert id(yasu) == id(y)
    else:
        assert id(yasu) != id(y)
