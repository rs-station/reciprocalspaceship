import gemmi
import numpy as np
import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize("sample_rate", [2.0, 3.0])
@pytest.mark.parametrize("p1", [True, False])
@pytest.mark.parametrize("use_sf", [True, False])
def test_to_reciprocalgrid_gemmi(mtz_by_spacegroup, sample_rate, p1, use_sf):
    """
    Test DataSet.to_reciprocalgrid() against gemmi for float data and complex
    structure factors
    """
    if p1:
        dataset = rs.read_mtz(mtz_by_spacegroup[:-4] + "_p1.mtz")
    else:
        dataset = rs.read_mtz(mtz_by_spacegroup)

    gemmimtz = dataset.to_gemmi()
    gridsize = rs.utils.get_gridsize(dataset, sample_rate=sample_rate)

    if use_sf:
        gemmigrid = gemmimtz.get_f_phi_on_grid("FMODEL", "PHIFMODEL", size=gridsize)
        expected = np.array(gemmigrid, copy=False)
        dataset["sf"] = dataset.to_structurefactor("FMODEL", "PHIFMODEL")
        result = dataset.to_reciprocalgrid("sf", gridsize=gridsize)

        # Requires rtol due to truncations applied in gemmi
        assert np.allclose(result, expected, rtol=1e-4)
    else:
        gemmigrid = gemmimtz.get_value_on_grid("FMODEL", size=gridsize)
        expected = np.array(gemmigrid, copy=False)
        result = dataset.to_reciprocalgrid("FMODEL", gridsize=gridsize)
        assert np.allclose(result, expected)


@pytest.mark.parametrize("sample_rate", [2.0, 3.0])
@pytest.mark.parametrize("column", ["FMODEL", "sf"])
def test_to_reciprocalgrid_p1(mtz_by_spacegroup, sample_rate, column):
    """
    DataSet.to_reciprocalgrid() should yield the same reciprocal grid if
    invoked with current spacegroup or the P1-ified version.
    """
    dataset = rs.read_mtz(mtz_by_spacegroup)
    dataset["sf"] = dataset.to_structurefactor("FMODEL", "PHIFMODEL")
    p1 = rs.read_mtz(mtz_by_spacegroup[:-4] + "_p1.mtz")
    p1["sf"] = p1.to_structurefactor("FMODEL", "PHIFMODEL")

    gridsize = rs.utils.get_gridsize(dataset, sample_rate=sample_rate)

    result1 = dataset.to_reciprocalgrid(column, gridsize=gridsize)
    result2 = p1.to_reciprocalgrid(column, gridsize=gridsize)
    assert np.allclose(result1, result2)


@pytest.mark.parametrize("sample_rate", [1.0, 2.5, 3.0])
@pytest.mark.parametrize("dmin", [10.0, 5.0, None])
@pytest.mark.parametrize("gridsize", [None, [144, 144, 144]])
def test_to_reciprocalgrid_sizes(mtz_by_spacegroup, sample_rate, dmin, gridsize):
    """
    Test DataSet.to_reciprocalgrid when invoked with different methods for specifying
    gridsize. If explicitly provided, `gridsize` should supersede `sample_rate` and `dmin`
    """
    dataset = rs.read_mtz(mtz_by_spacegroup)
    reciprocalgrid = dataset.to_reciprocalgrid(
        "FMODEL", sample_rate=sample_rate, dmin=dmin, gridsize=gridsize
    )

    if gridsize is None:
        expected = rs.utils.get_gridsize(dataset, sample_rate=sample_rate, dmin=dmin)
        assert reciprocalgrid.shape == tuple(expected)
    else:
        assert reciprocalgrid.shape == tuple(gridsize)
