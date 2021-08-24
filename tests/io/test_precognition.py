import gemmi
import pytest

import reciprocalspaceship as rs


def test_read_precognition_mtz(IOtest_mtz):
    """
    rs.read_precognition should raise ValueError when given file without
    .ii or .hkl suffix.
    """
    with pytest.raises(ValueError):
        rs.read_precognition(IOtest_mtz)


@pytest.mark.parametrize("spacegroup", [None, 96, "P 43 21 2"])
@pytest.mark.parametrize("cell", [None, (78.97, 78.97, 38.25, 90.0, 90.0, 90.0)])
def test_read_hkl(IOtest_hkl, spacegroup, cell):
    """
    Test rs.read_precognition() with a .hkl file
    """
    result = rs.read_precognition(IOtest_hkl, spacegroup=spacegroup, cell=cell)

    # Check main DataSet features
    assert isinstance(result, rs.DataSet)
    assert isinstance(result["F"], rs.DataSeries)
    assert result.columns.to_list() == ["F", "SigF"]
    assert list(result.index.names) == ["H", "K", "L"]

    # Check _metadata
    assert result._index_dtypes == {"H": "HKL", "K": "HKL", "L": "HKL"}
    if spacegroup:
        assert result.spacegroup.xhm() == gemmi.SpaceGroup(spacegroup).xhm()
    else:
        assert result.spacegroup is None
    if cell:
        assert result.cell.a == cell[0]
        assert result.cell.b == cell[1]
        assert result.cell.c == cell[2]
        assert result.cell.alpha == cell[3]
        assert result.cell.beta == cell[4]
        assert result.cell.gamma == cell[5]
    else:
        assert result.cell is None


@pytest.mark.parametrize("spacegroup", [None, 19, "P 21 21 21"])
@pytest.mark.parametrize("cell", [None, (35.0, 45.0, 99.0, 90.0, 90.0, 90.0)])
@pytest.mark.parametrize("log", [None, "log"])
def test_read_ii(IOtest_ii, IOtest_log, spacegroup, cell, log):
    """
    Test rs.read_precognition() with a .ii file
    """
    # Hacky way to parametrize a pytest fixture
    if log == "log":
        log = IOtest_log

    result = rs.read_precognition(
        IOtest_ii, spacegroup=spacegroup, cell=cell, logfile=log
    )

    # Check main DataSet features
    assert isinstance(result, rs.DataSet)
    assert isinstance(result["I"], rs.DataSeries)
    assert len(result.columns) == 7
    assert list(result.index.names) == ["H", "K", "L"]

    # Check _metadata
    assert result._index_dtypes == {"H": "HKL", "K": "HKL", "L": "HKL"}
    if spacegroup:
        assert result.spacegroup.xhm() == gemmi.SpaceGroup(spacegroup).xhm()
    else:
        assert result.spacegroup is None
    if log:
        assert result.cell.a == 34.4660
        assert result.cell.b == 45.6000
        assert result.cell.c == 99.5850
        assert result.cell.alpha == 90.0
        assert result.cell.beta == 90.0
        assert result.cell.gamma == 90.0
    elif cell:
        assert result.cell.a == cell[0]
        assert result.cell.b == cell[1]
        assert result.cell.c == cell[2]
        assert result.cell.alpha == cell[3]
        assert result.cell.beta == cell[4]
        assert result.cell.gamma == cell[5]
    else:
        assert result.cell is None
