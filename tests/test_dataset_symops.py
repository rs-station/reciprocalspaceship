import pytest
import numpy as np
from pandas.testing  import assert_index_equal
import reciprocalspaceship as rs

@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("reset_index", [True, False])
def test_hkl_to_asu(mtz_by_spacegroup, inplace, reset_index):
    """Test DataSet.hkl_to_asu() for common spacegroups"""
    x = rs.read_mtz(mtz_by_spacegroup)
    y = rs.read_mtz(mtz_by_spacegroup[:-4] + '_p1.mtz')
    y.spacegroup = x.spacegroup

    if reset_index:
        y.reset_index(inplace=True)

    yasu = y.hkl_to_asu(inplace=inplace)

    if reset_index:
        yasu.set_index(['H', 'K', 'L'], inplace=True)

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


def test_expand_to_p1(mtz_by_spacegroup):
    """Test DataSet.expand_to_p1() for common spacegroups"""
    x = rs.read_mtz(mtz_by_spacegroup)

    expected = rs.read_mtz(mtz_by_spacegroup[:-4] + '_p1.mtz')
    expected.sort_index(inplace=True)
    result = x.expand_to_p1()
    result.sort_index(inplace=True)

    expected_sf = expected.to_structurefactor("FMODEL", "PHIFMODEL")
    result_sf  = result.to_structurefactor("FMODEL", "PHIFMODEL")
    
    assert_index_equal(result.index, expected.index)
    assert np.allclose(result_sf.to_numpy(), expected_sf.to_numpy(), rtol=1e-4)


def test_expand_to_p1(mtz_by_spacegroup):
    """DataSet.expand_to_p1() should not affect P1 data"""
    expected = rs.read_mtz(mtz_by_spacegroup[:-4] + '_p1.mtz')
    result = expected.expand_to_p1()
    result.sort_index(inplace=True)
    expected.sort_index(inplace=True)    

    expected_sf = expected.to_structurefactor("FMODEL", "PHIFMODEL")
    result_sf  = result.to_structurefactor("FMODEL", "PHIFMODEL")
    
    assert_index_equal(result.index, expected.index)
    assert np.allclose(result_sf.to_numpy(), expected_sf.to_numpy(), rtol=1e-4)


def test_expand_to_p1_unmerged(data_unmerged):
    """Test DataSet.expand_to_p1() raises ValueError with unmerged data"""
    with pytest.raises(ValueError):
        result = data_unmerged.expand_to_p1()


def test_expand_anomalous(data_fmodel_P1):
    """
    Test DataSet.expand_anomalous() doubles reflections and applies 
    phase shifts
    """
    data_fmodel_P1["sf"] = data_fmodel_P1.to_structurefactor("FMODEL", "PHIFMODEL")
    friedel = data_fmodel_P1.expand_anomalous()

    assert len(friedel) == 2*len(data_fmodel_P1)

    H = data_fmodel_P1.get_hkls()
    assert np.array_equal(friedel.loc[H.tolist(), "FMODEL"].to_numpy(),
                          friedel.loc[(-1*H).tolist(), "FMODEL"].to_numpy())
    assert np.allclose(np.sin(np.deg2rad(friedel.loc[H.tolist(), "PHIFMODEL"].to_numpy())),
                       np.sin(np.deg2rad(-1*friedel.loc[(-1*H).tolist(), "PHIFMODEL"].to_numpy())), atol=1e-5)
    assert np.allclose(np.cos(np.deg2rad(friedel.loc[H.tolist(), "PHIFMODEL"].to_numpy())),
                       np.cos(np.deg2rad(-1*friedel.loc[(-1*H).tolist(), "PHIFMODEL"].to_numpy())), atol=1e-5)
    assert np.allclose(friedel.loc[H.tolist(), "sf"].to_numpy(),
                       np.conjugate(friedel.loc[(-1*H).tolist(), "sf"].to_numpy()))
