import pytest
from pandas.testing import assert_frame_equal

import reciprocalspaceship as rs
from reciprocalspaceship.utils import in_asu


@pytest.mark.parametrize(
    "plus_labels,minus_labels,suffixes",
    [
        (None, None, None),
        (None, None, ("(+)", "(-)")),
        ("I(+)", "I(-)", None),
        (["I(+)"], ["I(-)"], None),
        (["I(+)", "SIGI(+)"], ["I(-)", "SIGI(-)"], None),
        (("I(+)", "SIGI(+)"), ("I(-)", "SIGI(-)"), None),
    ],
)
def test_stack_anomalous(data_merged, plus_labels, minus_labels, suffixes):
    """
    Test behavior of DataSet.stack_anomalous()
    """
    if suffixes is None:
        result = data_merged.stack_anomalous(plus_labels, minus_labels)
    else:
        result = data_merged.stack_anomalous(plus_labels, minus_labels, suffixes)

    if plus_labels is None:
        if suffixes is not None:
            plus_labels = [l for l in data_merged.columns if l.endswith(suffixes[0])]
        else:
            plus_labels = [l for l in data_merged.columns if l.endswith("(+)")]
    elif isinstance(plus_labels, str):
        plus_labels = [plus_labels]

    assert len(result.columns) == (len(data_merged.columns) - len(plus_labels))

    centrics = data_merged.label_centrics()["CENTRIC"]
    assert len(result) == (2 * (~centrics).sum()) + centrics.sum()
    assert result.spacegroup.xhm() == data_merged.spacegroup.xhm()


@pytest.mark.parametrize(
    "plus_labels,minus_labels,suffixes",
    [
        (None, None, None),
        (None, None, 5),
        (None, None, ("(+)")),
        (None, None, ("(+)", "(-)", "(=)")),
        (None, "I(-)", None),
        (["I(+)", "SIGI(+)"], ["I(-)"], None),
        (["I(+)", "SIGI(+)"], ["SIGI(-)", "I(-)"], None),
    ],
)
def test_stack_anomalous_failure(data_merged, plus_labels, minus_labels, suffixes):
    """
    Test that DataSet.stack_anomalous() fails with improper arguments
    """
    with pytest.raises(ValueError):
        result = data_merged.stack_anomalous(plus_labels, minus_labels, suffixes)


@pytest.mark.parametrize(
    "label_dict,suffixes",
    [
        (
            {
                "I(+)": "Iplus",
                "SIGI(+)": "SIGIplus",
                "I(-)": "Iminus",
                "SIGI(-)": "SIGIminus",
            },
            ("plus", "minus"),
        ),
        (
            {"I(+)": "I+", "SIGI(+)": "SIGI+", "I(-)": "I-", "SIGI(-)": "SIGI-"},
            ("+", "-"),
        ),
    ],
)
def test_stack_anomalous_suffixes(data_merged, label_dict, suffixes):
    """
    Test DataSet.stack_anomalous() with custom suffixes
    """

    custom = data_merged.rename(columns=label_dict)
    result = custom.stack_anomalous(suffixes=suffixes)
    centrics = custom.label_centrics()["CENTRIC"]

    assert len(result) == (2 * (~centrics).sum()) + centrics.sum()
    assert result.spacegroup.xhm() == data_merged.spacegroup.xhm()


@pytest.mark.parametrize(
    "label_dict,suffixes",
    [
        (
            {
                "I(+)": "I(+)_foo",
                "SIGI(+)": "SIGI(+)_foo",
                "I(-)": "I(-)_foo",
                "SIGI(-)": "SIGI(-)_foo",
            },
            ("+", "-"),
        ),
    ],
)
def test_stack_anomalous_non_suffixes(data_merged, label_dict, suffixes):
    """
    Test DataSet.stack_anomalous() does not strip non-suffixes
    """

    custom = data_merged.rename(columns=label_dict)
    result = custom.stack_anomalous(suffixes=suffixes)

    assert "I" not in result.columns
    assert "I_foo" not in result.columns


def test_stack_anomalous_unmerged(data_unmerged):
    """
    Test DataSet.stack_anomalous() raises ValueError with unmerged data
    """
    with pytest.raises(ValueError):
        result = data_unmerged.stack_anomalous()


@pytest.mark.parametrize("columns", [None, "I", ["I", "SIGI"], ("I", "SIGI"), 5])
@pytest.mark.parametrize("suffixes", ["+-", ("(+)"), ("(+)", "(-)")])
def test_unstack_anomalous(data_merged, columns, suffixes):
    """Test behavior of DataSet.unstack_anomalous()"""

    data_merged = data_merged.stack_anomalous()
    newcolumns = columns

    def check_ValueError(data, columns, suffixes):
        with pytest.raises(ValueError):
            result = data.unstack_anomalous(columns, suffixes)
        return

    # Test input validation
    if columns is None:
        newcolumns = data_merged.columns.to_list()
    elif isinstance(columns, str):
        newcolumns = [columns]
    elif not isinstance(columns, (list, tuple)):
        check_ValueError(data_merged, columns, suffixes)
        return
    if not (isinstance(suffixes, (list, tuple)) and len(suffixes) == 2):
        check_ValueError(data_merged, columns, suffixes)
        return

    result = data_merged.unstack_anomalous(columns, suffixes)
    centrics = data_merged.label_centrics()["CENTRIC"]
    assert len(result.columns) == (len(data_merged.columns) + len(newcolumns))
    assert in_asu(result.get_hkls(), result.spacegroup).all()
    assert len(result) == ((~centrics).sum() / 2) + centrics.sum()
    assert result.spacegroup.xhm() == data_merged.spacegroup.xhm()


def test_unstack_anomalous_unmerged(data_unmerged):
    """
    Test DataSet.unstack_anomalous() raises ValueError with unmerged data
    """
    with pytest.raises(ValueError):
        result = data_unmerged.unstack_anomalous()


@pytest.mark.parametrize("rangeindexed", [True, False])
def test_roundtrip_stack_unstack_merged(data_merged, rangeindexed):
    """
    Test that DataSet is unchanged by roundtrip call of DataSet.stack_anomalous()
    followed by DataSet.unstack_anomalous()
    """
    if rangeindexed:
        data = data_merged.reset_index()
    else:
        data = data_merged

    stacked = data.stack_anomalous()
    result = stacked.unstack_anomalous(["I", "SIGI", "N"])

    # Re-order columns if needed
    result = result[data.columns]

    assert_frame_equal(result, data)
    assert result._index_dtypes == data._index_dtypes


@pytest.mark.parametrize("rangeindexed", [True, False])
def test_roundtrip_unstack_stack_merged(data_merged, rangeindexed):
    """
    Test that DataSet is unchanged by roundtrip call of DataSet.unstack_anomalous()
    followed by DataSet.stack_anomalous()
    """

    if rangeindexed:
        data = data_merged.reset_index()
    else:
        data = data_merged

    stacked = data.stack_anomalous()

    unstacked = stacked.unstack_anomalous(["I", "SIGI", "N"])
    result = unstacked.stack_anomalous()

    # Re-order columns if needed
    result = result[stacked.columns]

    assert_frame_equal(result, stacked, check_index_type=False)
    assert result._index_dtypes == stacked._index_dtypes
