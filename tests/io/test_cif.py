import filecmp
import tempfile
from os.path import exists

import gemmi
import pytest
from pandas.testing import assert_frame_equal

import reciprocalspaceship as rs
from reciprocalspaceship.utils import in_asu


def test_read_merged(IOtest_cif):
    """Test rs.read_cif() with merged SF CIF file"""
    dataset = rs.read_cif(IOtest_cif)

    assert dataset.spacegroup.number == 146
    assert dataset.columns.to_list() == ["FreeR_flag", "FC", "PHIC"]
    assert dataset.index.names == ["H", "K", "L"]
    assert dataset.merged
    assert isinstance(dataset, rs.DataSet)


def test_write_merged(IOtest_cif):
    """Test DataSet.write_mtz() with merged CIF file"""
    dataset = rs.read_cif(IOtest_cif)

    with tempfile.NamedTemporaryFile(suffix=".mtz") as temp:
        dataset.write_mtz(temp.name)
        assert exists(temp.name)


def test_write_merged_nosg(IOtest_cif):
    """Test that DataSet.write_mtz() without spacegroup raises AttributeError"""
    dataset = rs.read_cif(IOtest_cif)
    dataset.spacegroup = None

    with tempfile.NamedTemporaryFile(suffix=".mtz") as temp:
        with pytest.raises(AttributeError):
            dataset.write_mtz(temp.name)


def test_write_merged_nocell(IOtest_cif):
    """Test that DataSet.write_mtz() without cell raies AttributeError"""
    dataset = rs.read_cif(IOtest_cif)
    dataset.cell = None

    with tempfile.NamedTemporaryFile(suffix=".mtz") as temp:
        with pytest.raises(AttributeError):
            dataset.write_mtz(temp.name)


def test_roundtrip_merged(IOtest_cif):
    """Test roundtrip of rs.read_cif(), rs.read_mtz() and DataSet.write_mtz() with merged CIF file"""
    expected = rs.read_cif(IOtest_cif)

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


@pytest.mark.parametrize("project_name", [None, "project", "reciprocalspaceship", 1])
@pytest.mark.parametrize("crystal_name", [None, "crystal", "reciprocalspaceship", 1])
@pytest.mark.parametrize("dataset_name", [None, "dataset", "reciprocalspaceship", 1])
def test_to_gemmi_names(IOtest_cif, project_name, crystal_name, dataset_name):
    """
    Test that DataSet.to_gemmi() sets project/crystal/dataset names when given.

    ValueError should be raised for anything other than a string.
    """
    ds = rs.read_cif(IOtest_cif)

    if (
        not isinstance(project_name, str)
        or not isinstance(crystal_name, str)
        or not isinstance(dataset_name, str)
    ):
        with pytest.raises(ValueError):
            ds.to_gemmi(
                project_name=project_name,
                crystal_name=crystal_name,
                dataset_name=dataset_name,
            )
        return

    gemmimtz = ds.to_gemmi(
        project_name=project_name,
        crystal_name=crystal_name,
        dataset_name=dataset_name,
    )

    assert gemmimtz.dataset(0).project_name == project_name
    assert gemmimtz.dataset(0).crystal_name == crystal_name
    assert gemmimtz.dataset(0).dataset_name == dataset_name


@pytest.mark.parametrize("project_name", [None, "project", "reciprocalspaceship", 1])
@pytest.mark.parametrize("crystal_name", [None, "crystal", "reciprocalspaceship", 1])
@pytest.mark.parametrize("dataset_name", [None, "dataset", "reciprocalspaceship", 1])
def test_write_mtz_names(IOtest_cif, project_name, crystal_name, dataset_name):
    """
    Test that DataSet.write_mtz() sets project/crystal/dataset names when given.

    ValueError should be raised for anything other than a string.
    """
    ds = rs.read_cif(IOtest_cif)

    temp = tempfile.NamedTemporaryFile(suffix=".mtz")
    if (
        not isinstance(project_name, str)
        or not isinstance(crystal_name, str)
        or not isinstance(dataset_name, str)
    ):
        with pytest.raises(ValueError):
            ds.write_mtz(
                temp.name,
                project_name=project_name,
                crystal_name=crystal_name,
                dataset_name=dataset_name,
            )
        temp.close()
        return
    else:
        ds.write_mtz(
            temp.name,
            project_name=project_name,
            crystal_name=crystal_name,
            dataset_name=dataset_name,
        )

    gemmimtz = gemmi.read_mtz_file(temp.name)

    assert gemmimtz.dataset(0).project_name == project_name
    assert gemmimtz.dataset(0).crystal_name == crystal_name
    assert gemmimtz.dataset(0).dataset_name == dataset_name

    # Clean up
    temp.close()
