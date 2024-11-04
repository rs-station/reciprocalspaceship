import gemmi
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import reciprocalspaceship as rs


def test_constructor_empty():
    """Test DataSet.__init__()"""
    result = rs.DataSet()
    assert len(result) == 0
    assert result.spacegroup is None
    assert result.cell is None


@pytest.mark.parametrize("spacegroup", [None, gemmi.SpaceGroup(1), "P 1"])
@pytest.mark.parametrize(
    "cell", [None, gemmi.UnitCell(1, 1, 1, 90, 90, 90), (1, 1, 1, 90, 90, 90)]
)
@pytest.mark.parametrize("merged", [None, True, False])
def test_constructor_dataset(data_fmodel, spacegroup, cell, merged):
    """Test DataSet.__init__() when called with a DataSet"""
    result = rs.DataSet(data_fmodel, spacegroup=spacegroup, cell=cell, merged=merged)
    assert_frame_equal(result, data_fmodel)

    if merged is not None:
        assert result.merged == merged
    else:
        assert result.merged == True

    # Ensure provided values take precedence
    if spacegroup:
        assert result.spacegroup.xhm() == "P 1"
        assert isinstance(result.spacegroup, gemmi.SpaceGroup)
    else:
        assert result.spacegroup == data_fmodel.spacegroup

    if cell:
        if isinstance(cell, gemmi.UnitCell):
            assert result.cell == cell
        else:
            assert result.cell.a == cell[0]
        assert isinstance(result.cell, gemmi.UnitCell)
    else:
        assert result.cell == data_fmodel.cell


@pytest.mark.parametrize(
    "spacegroup", [None, 1, 5, 19, "P 21 21 21", "R 3:h", "R 3:blah", 1.2]
)
def test_spacegroup(data_fmodel, spacegroup):
    if spacegroup != 1.2 and spacegroup != "R 3:blah":
        data_fmodel.spacegroup = spacegroup
        if isinstance(spacegroup, (str, int)):
            assert data_fmodel.spacegroup.xhm() == gemmi.SpaceGroup(spacegroup).xhm()
        else:
            assert data_fmodel.spacegroup == spacegroup
    else:
        with pytest.raises(ValueError):
            data_fmodel.spacegroup = spacegroup


@pytest.mark.parametrize(
    "cell",
    [
        None,
        gemmi.UnitCell(10, 20, 30, 40, 50, 60),
        [10, 20, 30, 40, 50, 60],
        (10, 20, 30, 40, 50, 60),
        np.array([10, 20, 30, 40, 50, 60]),
        [10, 20, 30],
        [],
    ],
)
def test_cell(data_fmodel, cell):
    if cell is None:
        data_fmodel.cell = cell
        assert data_fmodel.cell is None
    elif not isinstance(cell, np.ndarray) and (cell == [] or cell == [10, 20, 30]):
        with pytest.raises(ValueError):
            data_fmodel.cell = cell
    else:
        data_fmodel.cell = cell
        if isinstance(cell, gemmi.UnitCell):
            expected = cell
        else:
            expected = gemmi.UnitCell(*cell)
        assert expected.a == data_fmodel.cell.a
        assert expected.b == data_fmodel.cell.b
        assert expected.c == data_fmodel.cell.c
        assert expected.alpha == data_fmodel.cell.alpha
        assert expected.beta == data_fmodel.cell.beta
        assert expected.gamma == data_fmodel.cell.gamma


@pytest.mark.parametrize("spacegroup", [None, gemmi.SpaceGroup(1)])
@pytest.mark.parametrize("cell", [None, gemmi.UnitCell(1, 1, 1, 90, 90, 90)])
def test_constructor_gemmi(data_gemmi, spacegroup, cell):
    """Test DataSet.__init__() when called with a DataSet"""
    result = rs.DataSet(data_gemmi, spacegroup=spacegroup, cell=cell)

    # Ensure provided values take precedence
    if spacegroup:
        assert result.spacegroup == spacegroup
    else:
        assert result.spacegroup == data_gemmi.spacegroup

    if cell:
        assert result.cell == cell
    else:
        assert result.cell == data_gemmi.cell


def test_to_structurefactor(data_fmodel):
    """Test DataSet.to_structurefactor()"""
    result = data_fmodel.to_structurefactor("FMODEL", "PHIFMODEL")
    sfamps = data_fmodel["FMODEL"].to_numpy()
    phases = data_fmodel["PHIFMODEL"].to_numpy()
    expected = sfamps * np.exp(1j * np.deg2rad(phases))
    assert isinstance(result, rs.DataSeries)
    assert np.allclose(result.to_numpy(), expected)


def test_from_structurefactor(data_fmodel):
    """Test DataSet.from_structurefactor()"""
    data_fmodel["sf"] = data_fmodel.to_structurefactor("FMODEL", "PHIFMODEL")
    f, phi = data_fmodel.from_structurefactor("sf")
    assert isinstance(f, rs.DataSeries)
    assert isinstance(phi, rs.DataSeries)
    assert isinstance(f.dtype, rs.StructureFactorAmplitudeDtype)
    assert isinstance(phi.dtype, rs.PhaseDtype)
    assert np.allclose(f.to_numpy(), data_fmodel["FMODEL"].to_numpy())
    assert np.allclose(
        np.sin(np.deg2rad(phi.to_numpy())),
        np.sin(np.deg2rad(data_fmodel["PHIFMODEL"].to_numpy())),
        atol=1e-6,
    )
    assert np.allclose(
        np.cos(np.deg2rad(phi.to_numpy())),
        np.cos(np.deg2rad(data_fmodel["PHIFMODEL"].to_numpy())),
        atol=1e-6,
    )


@pytest.mark.parametrize("skip_problem_mtztypes", [True, False])
def test_to_gemmi_roundtrip(data_gemmi, skip_problem_mtztypes):
    """Test DataSet.to_gemmi() and DataSet.from_gemmi() roundtrip"""
    rs_dataset = rs.DataSet.from_gemmi(data_gemmi)
    roundtrip = rs_dataset.to_gemmi(skip_problem_mtztypes)

    assert data_gemmi.spacegroup.number == roundtrip.spacegroup.number
    assert data_gemmi.cell.a == roundtrip.cell.a
    for c1, c2 in zip(data_gemmi.columns, roundtrip.columns):
        assert np.array_equal(c1.array, c2.array)
        assert c1.type == c2.type
        assert c1.label == c2.label


def test_from_gemmi(data_gemmi):
    """Test DataSet.from_gemmi()"""
    result = rs.DataSet.from_gemmi(data_gemmi)
    expected = rs.DataSet(data_gemmi)
    assert_frame_equal(result, expected)
    assert result.spacegroup == expected.spacegroup
    assert result.cell == expected.cell
    assert result._index_dtypes == expected._index_dtypes


def test_get_phase_keys(data_fmodel):
    """Test DataSet.get_phase_keys()"""
    result = data_fmodel.get_phase_keys()
    expected = "PHIFMODEL"
    assert len(result) == 1
    assert result[0] == expected

    # Add non-MTZDtype column
    data_fmodel.label_centrics(inplace=True)
    result = data_fmodel.get_phase_keys()
    assert len(result) == 1
    assert result[0] == expected


def test_get_m_isym_keys(data_fmodel):
    """Test DataSet.get_m_isym_keys()"""
    result = data_fmodel.get_m_isym_keys()
    assert result == []

    data_fmodel.hkl_to_asu(inplace=True)
    result = data_fmodel.get_m_isym_keys()
    assert len(result) == 1
    assert isinstance(result, list)
    assert isinstance(data_fmodel.dtypes[result[0]], rs.M_IsymDtype)


def test_get_hkls(data_fmodel):
    """Test DataSet.get_hkls()"""
    result = data_fmodel.get_hkls()
    expected = data_fmodel.reset_index()[["H", "K", "L"]].to_numpy()
    assert isinstance(result.flatten()[0], np.int32)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("no_sg", [True, False])
def test_label_centrics(data_fmodel, inplace, no_sg):
    """Test DataSet.label_centrics()"""
    if no_sg:
        data_fmodel.spacegroup = None
        with pytest.raises(ValueError):
            result = data_fmodel.label_centrics(inplace=inplace)
    else:
        result = data_fmodel.label_centrics(inplace=inplace)

        # Test inplace
        if inplace:
            assert id(result) == id(data_fmodel)
        else:
            assert id(result) != id(data_fmodel)

        # Test centric column
        assert "CENTRIC" in result
        assert result["CENTRIC"].dtype.name == "bool"


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("no_sg", [True, False])
def test_label_absences(data_fmodel, inplace, no_sg):
    """Test DataSet.label_absences()"""
    if no_sg:
        data_fmodel.spacegroup = None
        with pytest.raises(ValueError):
            result = data_fmodel.label_absences(inplace=inplace)
    else:
        result = data_fmodel.label_absences(inplace=inplace)

        # Test inplace
        if inplace:
            assert id(result) == id(data_fmodel)
        else:
            assert id(result) != id(data_fmodel)

        # Test centric column
        assert "ABSENT" in result
        assert result["ABSENT"].dtype.name == "bool"


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("hkl_index", [True, False])
def test_remove_absences(inplace, hkl_index):
    """Test DataSet.remove_absences()"""
    params = (34.0, 45.0, 98.0, 90.0, 90.0, 90.0)
    cell = gemmi.UnitCell(*params)
    sg_1 = gemmi.SpaceGroup(1)
    sg_19 = gemmi.SpaceGroup(19)
    Hall = rs.utils.generate_reciprocal_asu(cell, sg_1, 5.0, anomalous=False)
    h, k, l = Hall.T
    absent = rs.utils.is_absent(Hall, sg_19)
    ds = rs.DataSet(
        {
            "H": h,
            "K": k,
            "L": l,
            "I": np.ones(len(h)),
        },
        spacegroup=sg_19,
        cell=cell,
    ).infer_mtz_dtypes()
    if hkl_index:
        ds.set_index(["H", "K", "L"], inplace=True)

    ds_test = ds.remove_absences(inplace=inplace)
    ds_true = ds[~ds.label_absences().ABSENT]

    assert len(ds_test) == len(Hall) - absent.sum()
    assert np.array_equal(ds_test.get_hkls(), ds_true.get_hkls())

    if inplace:
        assert id(ds_test) == id(ds)
    else:
        assert id(ds_test) != id(ds)


@pytest.mark.parametrize("cache", [True, False])
def test_centrics(mtz_by_spacegroup, cache):
    """Test DataSet.centrics against DataSet.label_centrics"""
    ds = rs.read_mtz(mtz_by_spacegroup)
    if cache:
        ds.label_centrics(inplace=True)
        expected = ds.loc[ds.CENTRIC]
        result = ds.centrics
        assert "CENTRIC" in result
    else:
        expected = ds.loc[ds.label_centrics()["CENTRIC"]]
        result = ds.centrics
        assert "CENTRIC" not in result

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("cache", [True, False])
def test_acentrics(mtz_by_spacegroup, cache):
    """Test DataSet.acentrics against DataSet.label_centrics"""
    ds = rs.read_mtz(mtz_by_spacegroup)
    if cache:
        ds.label_centrics(inplace=True)
        expected = ds.loc[~ds.CENTRIC]
        result = ds.acentrics
        assert "CENTRIC" in result
    else:
        expected = ds.loc[~ds.label_centrics()["CENTRIC"]]
        result = ds.acentrics
        assert "CENTRIC" not in result

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("index", [True, False])
def test_infer_mtz_dtypes(data_merged, inplace, index):
    """Test DataSet.infer_mtz_dtypes()"""
    expected = data_merged
    temp = data_merged.astype(object, copy=False)
    result = temp.infer_mtz_dtypes(inplace=inplace, index=index)
    assert_frame_equal(result, expected)
    if inplace:
        assert id(result) == id(temp)
    else:
        assert id(result) != id(temp)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("index", [True, False])
def test_infer_mtz_dtypes_rangeindex(data_merged, inplace, index):
    """Test DataSet.infer_mtz_dtypes() with RangeIndex"""
    data_merged.reset_index(inplace=True)
    expected = data_merged
    temp = data_merged.astype(object, copy=False)
    result = temp.infer_mtz_dtypes(inplace=inplace, index=index)
    assert_frame_equal(result, expected)
    if inplace:
        assert id(result) == id(temp)
    else:
        assert id(result) != id(temp)


@pytest.mark.parametrize("inplace", [True, False])
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
def test_compute_dHKL(dataset_hkl, inplace, cell):
    """Test DataSet.compute_dHKL()"""
    dataset_hkl.cell = cell
    result = dataset_hkl.compute_dHKL(inplace=inplace)

    # Test inplace
    if inplace:
        assert id(result) == id(dataset_hkl)
    else:
        assert id(result) != id(dataset_hkl)

    # Compare to gemmi result
    expected = np.zeros(len(result), dtype=np.float32)
    for i, h in enumerate(result.get_hkls()):
        expected[i] = cell.calculate_d(h)
    assert np.allclose(result["dHKL"].to_numpy(), expected)
    assert isinstance(result["dHKL"].dtype, rs.MTZRealDtype)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("include_centering", [True, False])
@pytest.mark.parametrize(
    "spacegroup",
    [
        gemmi.SpaceGroup(19),
        gemmi.SpaceGroup(4),
    ],
)
def test_compute_multiplicity(dataset_hkl, inplace, include_centering, spacegroup):
    """Test DataSet.compute_multiplicity()"""
    dataset_hkl.spacegroup = spacegroup
    result = dataset_hkl.compute_multiplicity(
        inplace=inplace, include_centering=include_centering
    )

    # Test inplace
    if inplace:
        assert id(result) == id(dataset_hkl)
    else:
        assert id(result) != id(dataset_hkl)

    # Compare to gemmi result
    expected = np.zeros(len(result), dtype=np.int32)
    ops = spacegroup.operations()
    if include_centering:
        for i, h in enumerate(result.get_hkls()):
            expected[i] = ops.epsilon_factor(h)
    else:
        for i, h in enumerate(result.get_hkls()):
            expected[i] = ops.epsilon_factor_without_centering(h)

    assert np.all(result["EPSILON"].to_numpy() == expected)
    assert isinstance(result["EPSILON"].dtype, rs.MTZIntDtype)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize(
    "op",
    [
        "-x,-y,-z",
        gemmi.Op("x,y,z"),
        gemmi.Op("-x,-y,-z"),
        gemmi.Op("x,y,-z"),
        gemmi.Op("x,-y,-z"),
        gemmi.Op("-x,y,-z"),
        5,
    ],
)
def test_apply_symop_hkl(data_fmodel, inplace, op):
    """
    Test DataSet.apply_symop() using fmodel dataset. This test is purely
    for the HKL indices, but will not explicitly test phase shift for cases
    other than pure rotations.
    """

    copy = data_fmodel.copy()

    if isinstance(op, (gemmi.Op, str)):
        if isinstance(op, str):
            op = gemmi.Op(op)

        result = data_fmodel.apply_symop(op, inplace=inplace)
        expectedH = [op.apply_to_hkl(h) for h in copy.get_hkls()]
        expectedH = np.array(expectedH)

        assert np.array_equal(result["FMODEL"].to_numpy(), copy["FMODEL"].to_numpy())
        assert np.array_equal(result.get_hkls(), expectedH)

        # Confirm Miller indices are still correct dtype
        temp = result.reset_index()
        assert isinstance(temp.H.dtype, rs.HKLIndexDtype)
        assert isinstance(temp.K.dtype, rs.HKLIndexDtype)
        assert isinstance(temp.L.dtype, rs.HKLIndexDtype)

        # Confirm copy when desired
        if inplace:
            assert id(result) == id(data_fmodel)
        else:
            assert id(result) != id(data_fmodel)

    else:
        with pytest.raises(ValueError):
            result = data_fmodel.apply_symop(op, inplace=inplace)


@pytest.mark.parametrize("m_isym", [0, None, "M/ISYM", "I"])
def test_hklmapping_roundtrip(data_hewl, m_isym):
    """
    Test roundtrip of DataSet.hkl_to_asu() and DataSet.hkl_to_observed()
    """
    temp = data_hewl.hkl_to_asu()
    if m_isym == 0:
        with pytest.raises(ValueError):
            result = temp.hkl_to_observed(m_isym)
            return
    elif m_isym is not None and not m_isym in temp.columns:
        with pytest.raises(KeyError):
            result = temp.hkl_to_observed(m_isym)
            return
    elif m_isym == "I":
        with pytest.raises(ValueError):
            result = temp.hkl_to_observed(m_isym)
            return
    else:
        result = temp.hkl_to_observed(m_isym)
        result = result[data_hewl.columns]

        if data_hewl.merged:
            assert_frame_equal(result, data_hewl)
        else:
            pytest.xfail("DIALS M/ISYM column does not always use smallest ISYM value")


def test_hkl_to_observed_phase(data_fmodel_P1):
    """Test DataSet.hkl_to_observed() handling of phase"""
    data_fmodel_P1.spacegroup = gemmi.SpaceGroup(96)
    asu = data_fmodel_P1.hkl_to_asu()
    result = asu.hkl_to_observed()

    # Check phases have been canonicalized
    assert (result["PHIFMODEL"] >= -180.0).all()
    assert (result["PHIFMODEL"] <= 180.0).all()

    # Compare as complex structure factors
    original = rs.utils.to_structurefactor(
        data_fmodel_P1.FMODEL, data_fmodel_P1.PHIFMODEL
    )
    new = rs.utils.to_structurefactor(result.FMODEL, result.PHIFMODEL)
    assert np.allclose(new, original)


def test_hkl_to_observed_no_m_isym(data_fmodel_P1):
    """
    Test DataSet.hkl_to_observed() raises ValueError when DataSet has no
    M/ISYM columns
    """
    with pytest.raises(ValueError):
        data_fmodel_P1.hkl_to_observed()


def test_hkl_to_observed_2_m_isym(data_fmodel_P1):
    """
    Test DataSet.hkl_to_observed() raises ValueError when DataSet has 2
    M/ISYM columns
    """
    asu = data_fmodel_P1.hkl_to_asu()
    asu["EXTRA"] = 1
    asu["EXTRA"] = asu["EXTRA"].astype("M/ISYM")
    with pytest.raises(ValueError):
        data_fmodel_P1.hkl_to_observed()


def test_apply_symop_roundtrip(mtz_by_spacegroup):
    """
    Test DataSet.apply_symop() using fmodel datasets. This test will
    apply one of the symmetry operations, and confirm that hkl_to_asu()
    returns it to the same HKL, with the same phase
    """
    dataset = rs.read_mtz(mtz_by_spacegroup)
    for op in dataset.spacegroup.operations():
        applied = dataset.apply_symop(op)
        back = applied.apply_symop(op.inverse())

        assert np.array_equal(back.FMODEL.to_numpy(), dataset.FMODEL.to_numpy())
        assert np.array_equal(back.get_hkls(), dataset.get_hkls())

        original = rs.utils.to_structurefactor(dataset.FMODEL, dataset.PHIFMODEL)
        back = rs.utils.to_structurefactor(back.FMODEL, back.PHIFMODEL)
        assert np.isclose(original, back).all()


@pytest.mark.parametrize("inplace", [True, False])
def test_canonicalize_phases(data_fmodel, inplace):
    """Test DataSet.canonicalize_phases()"""
    temp = data_fmodel.copy()
    temp["PHIFMODEL"] += np.random.randint(-5, 5, len(temp)) * 360.0
    result = temp.canonicalize_phases(inplace=inplace)

    original = rs.utils.to_structurefactor(data_fmodel.FMODEL, data_fmodel.PHIFMODEL)
    new = rs.utils.to_structurefactor(result.FMODEL, result.PHIFMODEL)
    assert (result["PHIFMODEL"] >= -180.0).all()
    assert (result["PHIFMODEL"] <= 180.0).all()
    assert np.allclose(new, original)
    if inplace:
        assert id(result) == id(temp)
    else:
        assert id(result) != id(temp)


@pytest.mark.parametrize("sg1", [gemmi.SpaceGroup(96), None])
@pytest.mark.parametrize("sg2", [gemmi.SpaceGroup(96), gemmi.SpaceGroup(19), None])
@pytest.mark.parametrize("cell1", [gemmi.UnitCell(78.9, 78.9, 38.1, 90, 90, 90), None])
@pytest.mark.parametrize(
    "cell2",
    [
        gemmi.UnitCell(78.97, 78.97, 38.25, 90, 90, 90),
        gemmi.UnitCell(70.1, 70.1, 38.25, 90, 90, 90),
        None,
    ],
)
def test_is_isomorphous(data_unmerged, data_fmodel, sg1, sg2, cell1, cell2):
    """
    Test DataSet.is_isomorphous() using HEWL data and FMODEL mtz files.
    9LYZ is isomorphous, and should return True.
    """
    data_unmerged.spacegroup = sg1
    data_unmerged.cell = cell1
    data_fmodel.spacegroup = sg2
    data_fmodel.cell = cell2

    if (sg1 is None) or (sg2 is None) or (cell1 is None) or (cell2 is None):
        with pytest.raises(AttributeError):
            result = data_unmerged.is_isomorphous(data_fmodel)
    else:
        result = data_unmerged.is_isomorphous(data_fmodel)
        if (sg2.number == 96) and (cell2.a == 78.97):
            assert result
        else:
            assert not result


@pytest.mark.parametrize("threshold", [5.0, 1.0, 0.5, 0.1])
def test_is_isomorphous_threshold(threshold):
    """
    Test that DataSet.is_isorphous(self, other, cell_threshold) method's
    cell_threshold operates on percent difference.
    """
    epsilon = 1e-12
    cell = np.array([34.0, 45.0, 98.0, 90.0, 90.0, 90.0])
    spacegroup = 19

    ds = rs.DataSet(cell=cell, spacegroup=spacegroup)
    cell_resize_factor = (200.0 + threshold) / (200.0 - threshold)

    # Make a cell that should be exactly threshold percent bigger
    other_cell = cell_resize_factor * cell
    too_big_cell = other_cell + epsilon
    big_cell = other_cell - epsilon

    # Make a cell that should be exactly threshold percent smaller
    other_cell = cell / cell_resize_factor
    too_small_cell = other_cell - epsilon
    small_cell = other_cell + epsilon

    # Construct data sets
    too_big = rs.DataSet(cell=too_big_cell, spacegroup=spacegroup)
    big = rs.DataSet(cell=big_cell, spacegroup=spacegroup)
    too_small = rs.DataSet(cell=too_small_cell, spacegroup=spacegroup)
    small = rs.DataSet(cell=small_cell, spacegroup=spacegroup)

    # Cell is barely too big to be isomorphous
    assert not ds.is_isomorphous(too_big, threshold)

    # Cell is barely too small to be isomorphous
    assert not ds.is_isomorphous(too_small, threshold)

    # Cell is almost too big to be isomorphous
    assert ds.is_isomorphous(big, threshold)

    # Cell is almost too small to be isomorphous
    assert ds.is_isomorphous(small, threshold)


def test_to_gemmi_withNans(data_merged):
    """
    GH144: Test whether DataSet.to_gemmi() works with NaN-containing data.

    This test is intended to ensure DataSet works as intended with Phenix
    output, which often has many NaNs from filled reflections.
    """
    data_merged.loc[data_merged.index[0], "N(+)"] = np.nan
    roundtrip = rs.DataSet.from_gemmi(data_merged.to_gemmi())

    assert pd.isna(roundtrip.loc[data_merged.index[0], "N(+)"])


@pytest.mark.parametrize(
    "dtype",
    rs.summarize_mtz_dtypes(False)[["MTZ Code", "Name", "Class"]].melt().itertuples(),
)
@pytest.mark.parametrize("reset_index", [True, False])
def test_select_mtzdtype(data_hewl, dtype, reset_index):
    """
    Test DataSet.select_mtzdtype() with MTZDtype class, MTZ code, and dtype name
    """

    def _str_to_class(classname):
        """Convert string of class into instance of class"""
        import sys

        return getattr(sys.modules["reciprocalspaceship"], classname)

    # If True, remove HKL from index:
    if reset_index:
        data_hewl.reset_index(inplace=True)

    d = data_hewl.copy()
    if dtype.variable == "MTZ Code":
        dtype = dtype.value
        expected = d[
            [
                k
                for k in d
                if hasattr(d[k].dtype, "mtztype") and d[k].dtype.mtztype == dtype
            ]
        ]
        result = data_hewl.select_mtzdtype(dtype)
    elif dtype.variable == "Name":
        dtype = dtype.value
        expected = d[[k for k in d if d[k].dtype.name == dtype]]
        result = data_hewl.select_mtzdtype(dtype)
    else:
        dtype = _str_to_class(dtype.value)
        expected = d[[k for k in d if isinstance(d[k].dtype, dtype)]]
        result = data_hewl.select_mtzdtype(dtype())

    assert_frame_equal(result, expected)
    assert result.spacegroup.xhm() == data_hewl.spacegroup.xhm()
    assert np.array_equal(result.cell.parameters, data_hewl.cell.parameters)


@pytest.mark.parametrize("dtype", [1, 1.0, pd.Int32Dtype()])
def test_select_mtzdtype_ValueError(data_merged, dtype):
    """
    Test DataSet.select_mtzdtype() raises ValueError if `dtype` is not a
    string nor MTZDtype instance.
    """
    with pytest.raises(ValueError):
        data_merged.select_mtzdtype(dtype)
