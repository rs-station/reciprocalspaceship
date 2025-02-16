import gemmi
import numpy as np
import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize("sample_rate", [2.0, 3.0])
@pytest.mark.parametrize("p1", [True, False])
@pytest.mark.parametrize("use_sf", [True, False])
def test_to_reciprocal_grid_gemmi(mtz_by_spacegroup, sample_rate, p1, use_sf):
    """
    Test DataSet.to_reciprocal_grid() against gemmi for float data and complex
    structure factors
    """
    if p1:
        dataset = rs.read_mtz(mtz_by_spacegroup[:-4] + "_p1.mtz")
    else:
        dataset = rs.read_mtz(mtz_by_spacegroup)

    gemmimtz = dataset.to_gemmi()
    grid_size = dataset.get_reciprocal_grid_size(sample_rate=sample_rate)

    if use_sf:
        dataset["sf"] = dataset.to_structurefactor("FMODEL", "PHIFMODEL")
        result = dataset.to_reciprocal_grid("sf", grid_size=grid_size)
        gemmigrid = gemmimtz.get_f_phi_on_grid("FMODEL", "PHIFMODEL", size=grid_size)
        expected = gemmigrid.array

        # Requires rtol due to truncations applied in gemmi
        assert np.allclose(result, expected, rtol=1e-4)
    else:
        gemmigrid = gemmimtz.get_value_on_grid("FMODEL", size=grid_size)
        expected = np.array(gemmigrid, copy=False)
        result = dataset.to_reciprocal_grid("FMODEL", grid_size=grid_size)
        assert np.allclose(result, expected)


@pytest.mark.parametrize("sample_rate", [2.0, 3.0])
@pytest.mark.parametrize("column", ["FMODEL", "sf"])
def test_to_reciprocal_grid_p1(mtz_by_spacegroup, sample_rate, column):
    """
    DataSet.to_reciprocal_grid() should yield the same reciprocal grid if
    invoked with current spacegroup or the P1 version.
    """
    dataset = rs.read_mtz(mtz_by_spacegroup)
    dataset["sf"] = dataset.to_structurefactor("FMODEL", "PHIFMODEL")
    p1 = rs.read_mtz(mtz_by_spacegroup[:-4] + "_p1.mtz")
    p1["sf"] = p1.to_structurefactor("FMODEL", "PHIFMODEL")

    grid_size = dataset.get_reciprocal_grid_size(sample_rate=sample_rate)

    result1 = dataset.to_reciprocal_grid(column, grid_size=grid_size)
    result2 = p1.to_reciprocal_grid(column, grid_size=grid_size)
    assert np.allclose(result1, result2)


@pytest.mark.parametrize("sample_rate", [1.0, 2.5, 3.0])
@pytest.mark.parametrize("dmin", [10.0, 5.0, None])
@pytest.mark.parametrize("grid_size", [None, [144, 144, 144]])
def test_to_reciprocal_grid_sizes(mtz_by_spacegroup, sample_rate, dmin, grid_size):
    """
    Test DataSet.to_reciprocal_grid() when invoked with different methods for specifying
    grid size. If explicitly provided, `grid_size` should supersede `sample_rate` and `dmin`
    """
    dataset = rs.read_mtz(mtz_by_spacegroup)
    reciprocal_grid = dataset.to_reciprocal_grid(
        "FMODEL", sample_rate=sample_rate, dmin=dmin, grid_size=grid_size
    )

    if grid_size is None:
        expected = dataset.get_reciprocal_grid_size(dmin=dmin, sample_rate=sample_rate)
        assert reciprocal_grid.shape == tuple(expected)
    else:
        assert reciprocal_grid.shape == tuple(grid_size)


@pytest.mark.parametrize("sample_rate", [1.0, 2.5, 3.0])
@pytest.mark.parametrize("dmin", [10.0, 5.0, None])
@pytest.mark.parametrize("grid_size", [None, [144, 144, 144]])
def test_to_reciprocalgrid_sizes_DEPRECATED(
    mtz_by_spacegroup, sample_rate, dmin, grid_size
):
    """
    Test DataSet.to_reciprocalgrid() when invoked with different methods for specifying
    grid size. If explicitly provided, `grid_size` should supersede `sample_rate` and `dmin`

    The tested function is deprecated, and will be removed in a future release.
    """
    dataset = rs.read_mtz(mtz_by_spacegroup)

    with pytest.warns(DeprecationWarning):
        reciprocal_grid = dataset.to_reciprocalgrid(
            "FMODEL", sample_rate=sample_rate, dmin=dmin, gridsize=grid_size
        )

        if grid_size is None:
            expected = dataset.get_reciprocal_grid_size(
                dmin=dmin, sample_rate=sample_rate
            )
            assert reciprocal_grid.shape == tuple(expected)
        else:
            assert reciprocal_grid.shape == tuple(grid_size)
