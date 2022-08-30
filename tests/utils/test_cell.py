import gemmi
import numpy as np
import pytest

import reciprocalspaceship as rs
from reciprocalspaceship.utils import compute_dHKL, generate_reciprocal_cell


@pytest.mark.parametrize(
    "cell",
    [
        gemmi.UnitCell(
            10.0,
            20.0,
            30.0,
            90.0,
            90.0,
            90.0,
        ),
        gemmi.UnitCell(60.0, 60.0, 90.0, 90.0, 90.0, 120.0),
        gemmi.UnitCell(291.0, 423.0, 315.0, 90.0, 100.0, 90.0),
        gemmi.UnitCell(30.0, 50.0, 90.0, 75.0, 80.0, 106.0),
    ],
)
def test_compute_dHKL(dataset_hkl, cell):
    """Test rs.utils.compute_dHKL()"""
    hmax = 10
    H = (
        np.mgrid[-hmax : hmax + 1 : 1, -hmax : hmax + 1 : 1, -hmax : hmax + 1 : 1]
        .reshape((3, -1))
        .T
    )
    H = H[~np.all(H == 0, axis=1)]  # Remove 0,0,0
    result = compute_dHKL(H, cell)

    # Compare to gemmi result
    expected = np.zeros(len(result), dtype=np.float32)
    for i, h in enumerate(H):
        expected[i] = cell.calculate_d(h)

    assert np.allclose(result, expected)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("sample_rate", [0.5, 1.0, 2.5, 3.0])
@pytest.mark.parametrize("dmin", [10.0, 7.0, 5.0, 3.0, 2.0, None])
def test_get_gridsize(mtz_by_spacegroup, sample_rate, dmin):
    """
    Test rs.utils.get_gridsize() with fmodel data
    """
    dataset = rs.read_mtz(mtz_by_spacegroup)

    if sample_rate < 1.0:
        with pytest.raises(ValueError):
            rs.utils.get_gridsize(dataset, sample_rate=sample_rate, dmin=dmin)

    else:
        result = rs.utils.get_gridsize(dataset, sample_rate=sample_rate, dmin=dmin)

        assert isinstance(result, list)
        assert len(result) == 3

        if dmin is None:
            dmin = dataset.compute_dHKL().dHKL.min()

        # real-space grid spacing must be higher resolution than dmin/sample_rate
        abc = np.array(dataset.cell.parameters[:3])
        min_size = abc / (dmin / sample_rate)
        assert (np.array(result) >= min_size).all()


@pytest.mark.parametrize(
    "cell",
    [
        gemmi.UnitCell(
            10.0,
            20.0,
            30.0,
            90.0,
            90.0,
            90.0,
        ),
        gemmi.UnitCell(60.0, 60.0, 90.0, 90.0, 90.0, 120.0),
        gemmi.UnitCell(291.0, 423.0, 315.0, 90.0, 100.0, 90.0),
        gemmi.UnitCell(30.0, 50.0, 90.0, 75.0, 80.0, 106.0),
    ],
)
@pytest.mark.parametrize("dtype", [float, np.int32, int])
def test_generate_reciprocal_cell(cell, dtype):
    """Test rs.utils.generate_reciprocal_cell"""
    dmin = 5.0
    hkl = generate_reciprocal_cell(cell, dmin, dtype)

    assert hkl.dtype == dtype

    # Check that reflection 0,0,0 is omitted
    assert np.all(np.any(hkl != 0, axis=1))

    assert len(hkl) > 0
    assert len(np.unique(hkl, axis=0)) == len(hkl)
    assert compute_dHKL(hkl, cell).min() >= dmin
    assert cell.calculate_d_array(hkl).min() >= dmin
