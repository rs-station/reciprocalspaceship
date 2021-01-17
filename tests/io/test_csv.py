import pytest
import tempfile
import reciprocalspaceship as rs
import gemmi
from pandas.testing import assert_frame_equal

@pytest.mark.parametrize("sep", [",", "\t", ";"])
@pytest.mark.parametrize("spacegroup", [None, gemmi.SpaceGroup("P 1"), 1, "P 1"])
@pytest.mark.parametrize("cell", [
    None,
    gemmi.UnitCell(1, 1, 1, 90, 90, 90),
    (1, 1, 1, 90, 90, 90)
])
@pytest.mark.parametrize("merged", [None, True, False])
@pytest.mark.parametrize("infer_mtz_dtypes", [True, False])
def test_read_csv(IOtest_mtz, sep, spacegroup, cell, merged, infer_mtz_dtypes):
    """Test rs.read_csv()"""
    csvfile = tempfile.NamedTemporaryFile(suffix=".csv")
    expected = rs.read_mtz(IOtest_mtz)
    expected.to_csv(csvfile.name, sep=sep)
    result = rs.read_csv(csvfile.name, spacegroup=spacegroup, cell=cell,
                         merged=merged, infer_mtz_dtypes=infer_mtz_dtypes, sep=sep)
    print(result)
    result.set_index(["H", "K", "L"], inplace=True)
    csvfile.close()

    if spacegroup:
        if isinstance(spacegroup, gemmi.SpaceGroup):
            assert result.spacegroup.xhm() == spacegroup.xhm()
        else:
            assert result.spacegroup.xhm() == gemmi.SpaceGroup(spacegroup).xhm()
    else:
        assert result.spacegroup is None

    if cell:
        if isinstance(cell, gemmi.UnitCell):
            assert result.cell.a == cell.a
        else:
            assert result.cell.a == cell[0]
    else:
        assert result.cell is None

    assert result.merged == merged
    
    if infer_mtz_dtypes:
        assert_frame_equal(result, expected)
    else:
        assert_frame_equal(result, expected, check_dtype=False, check_index_type=False)
