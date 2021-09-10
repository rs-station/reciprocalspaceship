import tempfile

import gemmi
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import reciprocalspaceship as rs


@pytest.mark.parametrize("spacegroup", [None, 1, 4, 19])
@pytest.mark.parametrize(
    "cell",
    [None, gemmi.UnitCell(1, 1, 1, 90, 90, 90), gemmi.UnitCell(3, 4, 5, 90, 90, 120)],
)
@pytest.mark.parametrize("merged", [True, False])
def test_pickle_roundtrip_dataset(data_fmodel, spacegroup, cell, merged):
    """Test roundtrip of DataSet.to_pickle() and rs.read_pickle()"""
    pklfile = tempfile.NamedTemporaryFile(suffix=".pkl")

    # Setup expected result DataSet
    expected = data_fmodel
    expected.spacegroup = spacegroup
    expected.cell = cell
    expected.merged = merged

    # Roundtrip
    expected.to_pickle(pklfile.name)
    result = rs.read_pickle(pklfile.name)
    pklfile.close()

    assert isinstance(result, rs.DataSet)
    assert_frame_equal(result, expected)
    assert result._index_dtypes == expected._index_dtypes
    assert result.merged == merged

    if spacegroup:
        assert result.spacegroup.xhm() == expected.spacegroup.xhm()
    else:
        assert result.spacegroup is None

    if cell:
        assert result.cell.parameters == expected.cell.parameters
    else:
        assert result.cell is None


@pytest.mark.parametrize("label", ["FMODEL", "PHIFMODEL"])
def test_pickle_roundtrip_dataseries(data_fmodel, label):
    """Test roundtrip of DataSeries.to_pickle() and rs.read_pickle()"""
    pklfile = tempfile.NamedTemporaryFile(suffix=".pkl")

    # Setup expected result DataSeries
    expected = data_fmodel[label]

    # Roundtrip
    expected.to_pickle(pklfile.name)
    result = rs.read_pickle(pklfile.name)
    pklfile.close()

    assert isinstance(result, rs.DataSeries)
    assert_series_equal(result, expected, check_index_type=False)
