import pytest
import numpy as np
import reciprocalspaceship as rs
import gemmi
from pandas.testing import assert_frame_equal

def test_get_phase_keys(data_fmodel):
    """Test DataSet.get_phase_keys()"""
    result = data_fmodel.get_phase_keys()
    expected = "PHIFMODEL"
    assert len(result) == 1
    assert result[0] == expected

    # Add non-MTZDtype column
    data_fmodel.label_centrics(inplace=True)
    result = data_fmodel.get_phase_keys()
    assert len(result) == 1
    assert result[0] == expected
    
def test_get_hkls(data_hewl):
    """Test DataSet.get_hkls()"""
    H = data_hewl.get_hkls()
    assert H.shape == (len(data_hewl), 3)
    assert isinstance(H.flatten()[0], np.int32)
    pass

@pytest.mark.parametrize("inplace", [True, False])
def test_label_centrics(data_hewl, inplace):
    """Test DataSet.label_centrics()"""
    result = data_hewl.label_centrics(inplace=inplace)

    # Test inplace
    if inplace:
        assert id(result) == id(data_hewl)
    else:
        assert id(result) != id(data_hewl)

    # Test centric column
    assert "CENTRIC" in result
    assert result["CENTRIC"].dtype.name == "bool"
    
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("cell", [
    gemmi.UnitCell(10., 20., 30., 90., 90., 90.,),
    gemmi.UnitCell(60., 60., 90., 90., 90., 120.),
    gemmi.UnitCell(291., 423., 315., 90., 100., 90.)
])
def test_compute_dHKL(dataset_hkl, inplace, cell):
    """Test DataSet.compute_dHKL()"""
    dataset_hkl.cell = cell
    result = dataset_hkl.compute_dHKL(inplace=inplace)

    # Test inplace
    if inplace:
        assert id(result) == id(dataset_hkl)
    else:
        assert id(result) != id(dataset_hkl)
        
    # Compare to gemmi result
    expected = np.zeros(len(result), dtype=np.float32)
    for i, h in enumerate(result.get_hkls()):
        expected[i] = cell.calculate_d(h)
    assert np.allclose(result["dHKL"].to_numpy(), expected)
    assert isinstance(result["dHKL"].dtype, rs.MTZRealDtype)

@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("op", ["-x,-y,-z",
                                gemmi.Op("x,y,z"),
                                gemmi.Op("-x,-y,-z"),
                                gemmi.Op("x,y,-z"),
                                gemmi.Op("x,-y,-z"),
                                gemmi.Op("-x,y,-z"),
])
def test_apply_symop_hkl(data_fmodel, inplace, op):
    """
    Test DataSet.apply_symop() using fmodel dataset. This test is purely
    for the HKL indices, but will not explicitly test phase shift for cases
    other than pure rotations.
    """

    copy = data_fmodel.copy()

    if isinstance(op, gemmi.Op):

        result = data_fmodel.apply_symop(op, inplace=inplace)
        expectedH = [ op.apply_to_hkl(h) for h in copy.get_hkls() ]
        expectedH = np.array(expectedH)

        assert np.array_equal(result["FMODEL"].to_numpy(),
                              copy["FMODEL"].to_numpy())
        assert np.array_equal(result.get_hkls(), expectedH)

        # Confirm Miller indices are still correct dtype
        temp = result.reset_index()
        assert isinstance(temp.H.dtype, rs.HKLIndexDtype)
        assert isinstance(temp.K.dtype, rs.HKLIndexDtype)
        assert isinstance(temp.L.dtype, rs.HKLIndexDtype)        
        
        # Confirm copy when desired
        if inplace:
            assert id(result) == id(data_fmodel)
        else:
            assert id(result) != id(data_fmodel)

    else:
        with pytest.raises(ValueError):
            result = data_fmodel.apply_symop(op, inplace=inplace)
        
def test_apply_symop_roundtrip(mtz_by_spacegroup):
    """
    Test DataSet.apply_symop() using fmodel datasets. This test will
    apply one of the symmetry operations, and confirm that hkl_to_asu()
    returns it to the same HKL, with the same phase
    """
    dataset = rs.read_mtz(mtz_by_spacegroup)
    for op in dataset.spacegroup.operations():
        applied = dataset.apply_symop(op)
        back = applied.hkl_to_asu()

        assert np.array_equal(back.FMODEL.to_numpy(), dataset.FMODEL.to_numpy())
        assert np.array_equal(back.get_hkls(), dataset.get_hkls())

        original = rs.utils.to_structurefactor(dataset.FMODEL, dataset.PHIFMODEL)
        back = rs.utils.to_structurefactor(back.FMODEL, back.PHIFMODEL)
        assert np.isclose(original, back, rtol=1e-3).all()

def test_stack_anomalous_roundtrip(data_hewl):
    """
    Test that DataSet is unchanged by roundtrip call of DataSet.stack_anomalous()
    followed by DataSet.unstack_anomalous()
    """
    stacked = data_hewl.stack_anomalous()
    result = stacked.unstack_anomalous(["I", "SIGI", "N"])

    # Re-order columns if needed
    result = result[data_hewl.columns]

    assert_frame_equal(result, data_hewl)
