import pytest
import numpy as np
from reciprocalspaceship.utils import compute_dHKL, generate_reciprocal_cell
import gemmi


@pytest.mark.parametrize("cell", [
    gemmi.UnitCell(10., 20., 30., 90., 90., 90.,),
    gemmi.UnitCell(60., 60., 90., 90., 90., 120.),
    gemmi.UnitCell(291., 423., 315., 90., 100., 90.),
    gemmi.UnitCell(30., 50., 90., 75., 80., 106.),
])
def test_compute_dHKL(dataset_hkl, cell):
    """Test rs.utils.compute_dHKL()"""
    hmax = 10
    H = np.mgrid[-hmax:hmax+1:1,-hmax:hmax+1:1,-hmax:hmax+1:1].reshape((3, -1)).T
    H = H[~np.all(H==0, axis=1)] #Remove 0,0,0
    result = compute_dHKL(H, cell)

    # Compare to gemmi result
    expected = np.zeros(len(result), dtype=np.float32)
    for i, h in enumerate(H):
        expected[i] = cell.calculate_d(h)

    assert np.allclose(result, expected)
    assert np.all(np.isfinite(result))

@pytest.mark.parametrize("cell", [
    gemmi.UnitCell(10., 20., 30., 90., 90., 90.,),
    gemmi.UnitCell(60., 60., 90., 90., 90., 120.),
    gemmi.UnitCell(291., 423., 315., 90., 100., 90.),
    gemmi.UnitCell(30., 50., 90., 75., 80., 106.),
])
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

