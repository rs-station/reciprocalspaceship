import filecmp
import tempfile
from os.path import exists

import gemmi
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

import reciprocalspaceship as rs
from reciprocalspaceship.utils import in_asu


def test_read_merged(IOtest_mtz):
    """Test rs.read_mtz() with merged MTZ file"""
    dataset = rs.read_mtz(IOtest_mtz)

    assert dataset.spacegroup.number == 96
    assert dataset.columns.to_list() == ["FMODEL", "PHIFMODEL"]
    assert dataset.index.names == ["H", "K", "L"]
    assert isinstance(dataset, rs.DataSet)


def test_write_merged(IOtest_mtz):
    """Test DataSet.write_mtz() with merged MTZ file"""
    dataset = rs.read_mtz(IOtest_mtz)

    with tempfile.NamedTemporaryFile(suffix=".mtz") as temp:
        dataset.write_mtz(temp.name)
        assert exists(temp.name)


def test_write_merged_nosg(IOtest_mtz):
    """Test that DataSet.write_mtz() without spacegroup raises AttributeError"""
    dataset = rs.read_mtz(IOtest_mtz)
    dataset.spacegroup = None

    with tempfile.NamedTemporaryFile(suffix=".mtz") as temp:
        with pytest.raises(AttributeError):
            dataset.write_mtz(temp.name)


def test_write_merged_nocell(IOtest_mtz):
    """Test that DataSet.write_mtz() without cell raises AttributeError"""
    dataset = rs.read_mtz(IOtest_mtz)
    dataset.cell = None

    with tempfile.NamedTemporaryFile(suffix=".mtz") as temp:
        with pytest.raises(AttributeError):
            dataset.write_mtz(temp.name)


@pytest.mark.parametrize("skip_problem_mtztypes", [True, False])
def test_write_merged_nocell(IOtest_mtz, skip_problem_mtztypes):
    """
    Test skip_problem_mtztypes flag of DataSet.write_mtz()
    """
    dataset = rs.read_mtz(IOtest_mtz)
    dataset["nonMTZ"] = 1

    with tempfile.NamedTemporaryFile(suffix=".mtz") as temp:
        if not skip_problem_mtztypes:
            with pytest.raises(ValueError):
                dataset.write_mtz(
                    temp.name, skip_problem_mtztypes=skip_problem_mtztypes
                )
        else:
            dataset.write_mtz(temp.name, skip_problem_mtztypes=skip_problem_mtztypes)
            assert exists(temp.name)


def test_roundtrip_merged(IOtest_mtz):
    """Test roundtrip of rs.read_mtz() and DataSet.write_mtz() with merged MTZ file"""
    expected = rs.read_mtz(IOtest_mtz)

    temp1 = tempfile.NamedTemporaryFile(suffix=".mtz")
    temp2 = tempfile.NamedTemporaryFile(suffix=".mtz")

    expected.write_mtz(temp1.name)
    result = rs.read_mtz(temp1.name)
    result.write_mtz(temp2.name)

    assert_frame_equal(result, expected)
    assert filecmp.cmp(temp1.name, temp2.name)

    # Clean up
    temp1.close()
    temp2.close()


def test_read_unmerged(data_unmerged):
    """Test rs.read_mtz() with unmerged data"""
    # Unmerged data will not be in asu, and should have a PARTIAL column
    assert not in_asu(data_unmerged.get_hkls(), data_unmerged.spacegroup).all()
    assert "PARTIAL" in data_unmerged.columns
    assert data_unmerged["PARTIAL"].dtype.name == "bool"
    assert not "M/ISYM" in data_unmerged.columns
    assert not data_unmerged.merged


def test_read_unmerged_2m_isym(data_unmerged):
    """Test rs.read_mtz() with unmerged data containing 2 M/ISYM columns"""
    data_unmerged["EXTRA"] = 1
    data_unmerged["EXTRA"] = data_unmerged["EXTRA"].astype("M/ISYM")
    temp = tempfile.NamedTemporaryFile(suffix=".mtz")
    data_unmerged.write_mtz(temp.name)
    with pytest.raises(ValueError):
        fails = rs.read_mtz(temp.name)

    # Clean up
    temp.close()


@pytest.mark.parametrize("label_centrics", [True, False])
def test_roundtrip_unmerged(data_unmerged, label_centrics):
    """
    Test roundtrip of rs.read_mtz() and DataSet.write_mtz() with unmerged data
    """
    if label_centrics:
        data_unmerged.label_centrics(inplace=True)

    temp = tempfile.NamedTemporaryFile(suffix=".mtz")
    temp2 = tempfile.NamedTemporaryFile(suffix=".mtz")
    data_unmerged.write_mtz(temp.name)
    data2 = rs.read_mtz(temp.name)
    data2 = data2[data_unmerged.columns]  # Ensure consistent column ordering
    data2.write_mtz(temp2.name)
    assert filecmp.cmp(temp.name, temp2.name)
    assert_frame_equal(data_unmerged, data2)
    assert data_unmerged.merged == data2.merged

    # Clean up
    temp.close()
    temp2.close()


def test_unmerged_after_write(data_unmerged):
    """
    #110: Test that unmerged DataSet objects are unchanged following calls to
    DataSet.write_mtz()
    """
    expected = data_unmerged.copy()
    data_unmerged.write_mtz("/dev/null")
    assert_frame_equal(data_unmerged, expected)
