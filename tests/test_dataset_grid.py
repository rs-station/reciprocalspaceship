import pytest
import numpy as np
import gemmi
import reciprocalspaceship as rs

@pytest.mark.parametrize("sample_rate", [1, 2, 3])
@pytest.mark.parametrize("p1", [True, False])
def test_to_reciprocalgrid_complex(mtz_by_spacegroup, sample_rate, p1):
    """
    Test DataSet.to_reciprocalgrid() against gemmi for complex  data
    """
    if p1:
        dataset = rs.read_mtz(mtz_by_spacegroup[:-4] + "_p1.mtz")
    else:
        dataset = rs.read_mtz(mtz_by_spacegroup)
    gemmimtz = dataset.to_gemmi()

    # Note: Use P1 data to determine proper gridsize    
    testp1 = dataset.expand_to_p1()
    testp1.spacegroup = dataset.spacegroup
    gridsize = testp1.to_gemmi().get_size_for_hkl(sample_rate=sample_rate)

    gemmigrid = gemmimtz.get_f_phi_on_grid("FMODEL", "PHIFMODEL", size=gridsize)
    expected = np.array(gemmigrid)
    dataset["sf"] = dataset.to_structurefactor("FMODEL", "PHIFMODEL")
    result = dataset.to_reciprocalgrid("sf", gridsize)
    assert np.allclose(result, expected, rtol=1e-4)

@pytest.mark.parametrize("sample_rate", [1, 2, 3])
@pytest.mark.parametrize("p1", [True, False])
def test_to_reciprocalgrid_float(mtz_by_spacegroup, sample_rate, p1):
    """
    Test DataSet.to_reciprocalgrid() against gemmi for float data
    """
    if p1:
        dataset = rs.read_mtz(mtz_by_spacegroup[:-4] + "_p1.mtz")
    else:
        dataset = rs.read_mtz(mtz_by_spacegroup)
    gemmimtz = dataset.to_gemmi()

    # Note: Use P1 data to determine proper gridsize
    testp1 = dataset.expand_to_p1()
    testp1.spacegroup = dataset.spacegroup
    gridsize = testp1.to_gemmi().get_size_for_hkl(sample_rate=sample_rate)

    gemmigrid = gemmimtz.get_value_on_grid("FMODEL", size=gridsize)
    expected = np.array(gemmigrid)
    result = dataset.to_reciprocalgrid("FMODEL", gridsize)

    assert np.allclose(result, expected)

