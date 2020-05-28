import pytest
import numpy as np
import reciprocalspaceship as rs
import gemmi

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

