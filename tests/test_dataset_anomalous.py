import pytest
from pandas.testing import assert_frame_equal
import reciprocalspaceship as rs
from reciprocalspaceship.utils import in_asu

@pytest.mark.parametrize("labels", [
    (None, None),
    ("I(+)", "I(-)"),
    (["I(+)",  "SIGI(+)"], ["I(-)", "SIGI(-)"]),
    (("I(+)",  "SIGI(+)"), ("I(-)", "SIGI(-)")),
    (None, "I(-)"),
    (["I(+)", "SIGI(+)"], ["I(-)"]),
    (["I(+)", "SIGI(+)"], ["SIGI(-)", "I(-)"]),    
])
def test_stack_anomalous(data_hewl, labels):
    """Test behavior of DataSet.stack_anomalous()"""

    plus_labels = labels[0]
    minus_labels = labels[1]
    
    if not data_hewl.merged:
        data_hewl = data_hewl.unstack_anomalous(["I", "SIGI"])

    # Check input data
    def check_ValueError(data, plus_labels, minus_labels):
        with pytest.raises(ValueError):
            result = data.stack_anomalous(plus_labels, minus_labels)
        return

    if (plus_labels is None and minus_labels is None):
        plus_labels  = [ l for l in data_hewl.columns if "(+)" in l ]
        minus_labels = [ l for l in data_hewl.columns if "(-)" in l ]

    if isinstance(plus_labels, str) and isinstance(minus_labels, str):
        plus_labels = [plus_labels]
        minus_labels =[minus_labels]
    elif (isinstance(plus_labels, list) and
          isinstance(minus_labels, list)):
        if len(plus_labels) != len(minus_labels):
            check_ValueError(data_hewl, plus_labels, minus_labels)
            return
    else:
        check_ValueError(data_hewl, plus_labels, minus_labels)
        return

    for plus, minus in zip(plus_labels, minus_labels):
        if data_hewl[plus].dtype != data_hewl[minus].dtype:
            check_ValueError(data_hewl, plus_labels, minus_labels)
            return

    result = data_hewl.stack_anomalous(plus_labels, minus_labels)
    assert len(result.columns) == (len(data_hewl.columns)-len(plus_labels))

@pytest.mark.parametrize("columns", [
    None,
    "I",
    ["I", "SIGI"],
    ("I", "SIGI"),
    5
])
@pytest.mark.parametrize("suffixes", [
    "+-",
    ("(+)"),
    ("(+)", "(-)")
])
def test_unstack_anomalous(data_hewl, columns, suffixes):
    """Test behavior of DataSet.unstack_anomalous()"""

    if data_hewl.merged:
        data_hewl = data_hewl.stack_anomalous()
    
    def check_ValueError(data, columns, suffixes):
        with pytest.raises(ValueError):
            result = data.unstack_anomalous(columns, suffixes)
        return
    
    # Test input validation
    if columns is None:
        columns = data_hewl.columns.to_list()
    elif isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, (list, tuple)):
        check_ValueError(data_hewl, columns, suffixes)
        return
    if not (isinstance(suffixes, (list, tuple)) and len(suffixes) == 2):
        check_ValueError(data_hewl, columns, suffixes)
        return

    result = data_hewl.unstack_anomalous(columns, suffixes)
    assert len(result.columns) == (len(data_hewl.columns)+len(columns))
    assert in_asu(result.get_hkls(), result.spacegroup).all()
    
    # Check expected behavior with merged DataSet
    if data_hewl.merged:
        assert len(result) == len(data_hewl)/2

    # Check expected behavior with unmerged DataSet
    else:
        assert len(result) == len(data_hewl)
        
def test_roundtrip_merged(data_merged):
    """
    Test that DataSet is unchanged by roundtrip call of DataSet.stack_anomalous()
    followed by DataSet.unstack_anomalous()
    """
    stacked = data_merged.stack_anomalous()
    result = stacked.unstack_anomalous(["I", "SIGI", "N"])

    # Re-order columns if needed
    result = result[data_merged.columns]

    assert_frame_equal(result, data_merged)


def test_roundtrip_unmerged(data_unmerged):
    """
    Test that DataSet is unchanged by roundtrip call of DataSet.unstack_anomalous()
    followed by DataSet.stack_anomalous()
    """
    unstacked = data_unmerged.unstack_anomalous(["I", "SIGI"])
    result = unstacked.stack_anomalous()

    # Re-order columns if needed
    result = result[data_unmerged.columns]
    result.reset_index(inplace=True)
    result.set_index(["H", "K", "L", "BATCH"], inplace=True)
    data_unmerged.set_index(["BATCH"], append=True, inplace=True)
    
    # Sort indices for comparison
    result.sort_index(inplace=True)
    data_unmerged.sort_index(inplace=True)

    assert_frame_equal(result, data_unmerged)


