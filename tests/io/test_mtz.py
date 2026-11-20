import filecmp
import tempfile
import warnings
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
def test_write_merged_nonMTZDtype(IOtest_mtz, skip_problem_mtztypes):
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


@pytest.mark.parametrize("in_asu", [True, False])
def test_unmerged_after_write(data_unmerged, in_asu):
    """
    #110: Test that unmerged DataSet objects are unchanged following calls to
    DataSet.write_mtz()
    """
    if in_asu:
        data_unmerged.hkl_to_asu(inplace=True)
    expected = data_unmerged.copy()
    data_unmerged.write_mtz("/dev/null")
    assert_frame_equal(data_unmerged, expected)


@pytest.mark.parametrize("project_name", [None, "project", "reciprocalspaceship", 1])
@pytest.mark.parametrize("crystal_name", [None, "crystal", "reciprocalspaceship", 1])
@pytest.mark.parametrize("dataset_name", [None, "dataset", "reciprocalspaceship", 1])
def test_to_gemmi_names(IOtest_mtz, project_name, crystal_name, dataset_name):
    """
    Test that DataSet.to_gemmi() sets project/crystal/dataset names when given.

    ValueError should be raised for anything other than a string.
    """
    ds = rs.read_mtz(IOtest_mtz)

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
def test_write_mtz_names(IOtest_mtz, project_name, crystal_name, dataset_name):
    """
    Test that DataSet.write_mtz() sets project/crystal/dataset names when given.

    ValueError should be raised for anything other than a string.
    """
    ds = rs.read_mtz(IOtest_mtz)

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


@pytest.mark.parametrize("range_indexed", [True, False])
def test_write_mtz_rangeindexed(data_hewl, range_indexed):
    """
    GH#255: Test DataSet.write_mtz() with pd.RangeIndex index.
    """
    # Add centrics column to test relevant code paths
    data_hewl.label_centrics(inplace=True)

    if range_indexed:
        data_hewl = data_hewl.reset_index()

    data_before = data_hewl.copy()

    with tempfile.NamedTemporaryFile(suffix=".mtz") as temp:
        data_hewl.write_mtz(temp.name)
        assert exists(temp.name)

    # Make sure writing MTZ did not alter calling DataSet
    assert_frame_equal(data_hewl, data_before, check_like=True)


@pytest.mark.parametrize("range_indexed", [True, False])
def test_to_gemmi_rangeindexed(data_hewl, range_indexed):
    """
    GH#255: Test DataSet.to_gemmi() with pd.RangeIndex index.
    """
    # Add centrics column to test relevant code paths
    data_hewl.label_centrics(inplace=True)

    if range_indexed:
        data_hewl = data_hewl.reset_index()

    data_before = data_hewl.copy()

    mtz = data_hewl.to_gemmi()
    data_roundtrip = rs.DataSet.from_gemmi(mtz)
    if range_indexed:
        data_roundtrip = data_roundtrip.reset_index()

    assert_frame_equal(data_hewl, data_roundtrip)

    # Make sure writing MTZ did not alter calling DataSet
    assert_frame_equal(data_hewl, data_before, check_like=True)


# --- Multi-dataset MTZ files (GH#336) ---------------------------------------


def _make_multi_dataset_mtz(datasets, unmerged=False):
    """
    Construct a gemmi.Mtz with an ``HKL_base`` dataset holding only H, K, L
    followed by the given datasets. This mirrors the layout produced by
    XDS -> XDSCONV -> CAD and autoPROC/STARANISO (see GH#336).

    Parameters
    ----------
    datasets : list of (str, float, list of (label, mtztype))
        dataset name, wavelength, and the columns belonging to that dataset
    unmerged : bool
        If True, write repeated Miller indices and an M/ISYM column in the
        last dataset.
    """
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.SpaceGroup("P 31 2 1")
    mtz.set_cell_for_all(gemmi.UnitCell(83.617, 83.617, 84.078, 90, 90, 120))
    mtz.datasets[0].wavelength = 9.999

    labels = []
    for i, (name, wavelength, columns) in enumerate(datasets, start=1):
        ds = mtz.add_dataset(name)
        ds.project_name = "Test"
        ds.crystal_name = "A"
        ds.wavelength = wavelength
        for label, mtztype in columns:
            mtz.add_column(label, mtztype, dataset_id=ds.id)
            labels.append(label)
    if unmerged:
        mtz.add_column("M/ISYM", "Y", dataset_id=mtz.datasets[-1].id)
        labels.append("M/ISYM")

    hkls = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1], [2, 0, 1], [1, 2, 3]])
    if unmerged:
        hkls = np.concatenate([hkls, hkls])
    data = np.zeros((len(hkls), 3 + len(labels)), dtype="float32")
    data[:, :3] = hkls
    data[:, 3 : 3 + len(labels)] = np.arange(1, len(labels) + 1)
    if unmerged:
        data[:, -1] = 1.0
    mtz.set_data(data)
    return mtz


@pytest.fixture
def mtz_hkl_base(tmp_path):
    """Two-dataset MTZ: HKL_base (wavelength 9.999) + one data dataset"""
    mtz = _make_multi_dataset_mtz(
        [("1", 1.05965, [("F", "F"), ("SIGF", "Q"), ("IMEAN", "J")])]
    )
    path = str(tmp_path / "hkl_base.mtz")
    mtz.write_to_file(path)
    return path


@pytest.fixture
def mtz_mad(tmp_path):
    """Three-dataset MTZ: HKL_base + two datasets with different wavelengths"""
    mtz = _make_multi_dataset_mtz(
        [
            ("peak", 0.9792, [("F_peak", "F"), ("SIGF_peak", "Q")]),
            ("remote", 0.9000, [("F_remote", "F"), ("SIGF_remote", "Q")]),
        ]
    )
    path = str(tmp_path / "mad.mtz")
    mtz.write_to_file(path)
    return path


def test_read_mtz_multiple_default_is_false(IOtest_mtz):
    """multiple=False returns exactly what the default call returns"""
    expected = rs.read_mtz(IOtest_mtz)
    result = rs.read_mtz(IOtest_mtz, multiple=False)
    assert isinstance(result, rs.DataSet)
    assert_frame_equal(result, expected)
    assert result.wavelength == expected.wavelength
    assert result.merged == expected.merged


def test_read_mtz_multiple_single_dataset(IOtest_mtz):
    """A single-dataset MTZ yields one DataSet identical to the default read"""
    expected = rs.read_mtz(IOtest_mtz)
    result = rs.read_mtz(IOtest_mtz, multiple=True)
    assert not isinstance(result, rs.DataSet)
    result = list(result)
    assert len(result) == 1
    assert_frame_equal(result[0], expected)
    assert result[0].wavelength == expected.wavelength
    assert result[0].spacegroup.xhm() == expected.spacegroup.xhm()
    assert result[0].cell.parameters == expected.cell.parameters


def test_read_mtz_hkl_base_default(mtz_hkl_base):
    """
    GH#336: the placeholder wavelength of an HKL_base dataset must not be
    used, and must not trigger the multiple-wavelength warning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds = rs.read_mtz(mtz_hkl_base)

    assert ds.wavelength == pytest.approx(1.05965)
    assert ds.columns.to_list() == ["F", "SIGF", "IMEAN"]
    assert ds.index.names == ["H", "K", "L"]
    assert ds.merged


def test_read_mtz_hkl_base_multiple(mtz_hkl_base):
    """Every dataset is yielded, including the H/K/L-only HKL_base"""
    datasets = list(rs.read_mtz(mtz_hkl_base, multiple=True))
    assert len(datasets) == 2

    base, data = datasets
    assert base.columns.to_list() == []
    assert base.index.names == ["H", "K", "L"]
    assert base.wavelength == pytest.approx(9.999)
    assert len(base) == 5

    assert data.columns.to_list() == ["F", "SIGF", "IMEAN"]
    assert data.index.names == ["H", "K", "L"]
    assert data.wavelength == pytest.approx(1.05965)
    assert_frame_equal(data, rs.read_mtz(mtz_hkl_base))

    for ds in datasets:
        assert isinstance(ds, rs.DataSet)
        assert ds.spacegroup.number == 152
        assert ds.cell.a == pytest.approx(83.617)
        assert ds.merged


def test_read_mtz_mad_default_warns(mtz_mad):
    """Datasets with genuinely different wavelengths still warn by default"""
    with pytest.warns(UserWarning, match="multiple=True"):
        ds = rs.read_mtz(mtz_mad)
    assert ds.wavelength == pytest.approx(0.9792)
    assert ds.columns.to_list() == ["F_peak", "SIGF_peak", "F_remote", "SIGF_remote"]


def test_read_mtz_mad_multiple(mtz_mad):
    """Each dataset is yielded with its own columns and wavelength"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        datasets = list(rs.read_mtz(mtz_mad, multiple=True))

    assert [ds.columns.to_list() for ds in datasets] == [
        [],
        ["F_peak", "SIGF_peak"],
        ["F_remote", "SIGF_remote"],
    ]
    assert [ds.wavelength for ds in datasets] == pytest.approx([9.999, 0.9792, 0.9])

    combined = rs.read_mtz(mtz_mad)
    for ds in datasets[1:]:
        assert_frame_equal(ds, combined[ds.columns.to_list()])


def test_read_mtz_multiple_is_iterator(mtz_mad):
    """multiple=True returns a lazy iterator"""
    result = rs.read_mtz(mtz_mad, multiple=True)
    assert iter(result) is result
    first = next(result)
    assert first.columns.to_list() == []
    assert len(list(result)) == 2


def test_read_mtz_multiple_unmerged(tmp_path):
    """Merged/unmerged status is detected for each yielded DataSet"""
    mtz = _make_multi_dataset_mtz(
        [("1", 1.0, [("I", "J"), ("SIGI", "Q")])], unmerged=True
    )
    path = str(tmp_path / "unmerged.mtz")
    mtz.write_to_file(path)

    expected = rs.read_mtz(path)
    assert not expected.merged

    base, data = rs.read_mtz(path, multiple=True)
    assert base.merged
    assert not data.merged
    assert "PARTIAL" in data.columns
    assert_frame_equal(data, expected)


def test_read_mtz_multiple_missing_hkl(tmp_path):
    """A missing Miller index column raises before iteration begins"""
    mtz = gemmi.Mtz()
    mtz.spacegroup = gemmi.SpaceGroup("P 1")
    mtz.cell = gemmi.UnitCell(10, 10, 10, 90, 90, 90)
    mtz.add_dataset("d")
    mtz.add_column("H", "H")
    mtz.add_column("K", "H")
    mtz.add_column("F", "F")
    mtz.set_data(np.ones((3, 3), dtype="float32"))
    path = str(tmp_path / "no_l.mtz")
    mtz.write_to_file(path)

    with pytest.raises(ValueError, match="H, K, L"):
        rs.read_mtz(path, multiple=True)
