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
def test_stack_anomalous(data_merged, labels):
    """Test behavior of DataSet.stack_anomalous()"""

    plus_labels = labels[0]
    minus_labels = labels[1]
    
    # Check input data
    def check_ValueError(data, plus_labels, minus_labels):
        with pytest.raises(ValueError):
            result = data.stack_anomalous(plus_labels, minus_labels)
        return

    if (plus_labels is None and minus_labels is None):
        plus_labels  = [ l for l in data_merged.columns if "(+)" in l ]
        minus_labels = [ l for l in data_merged.columns if "(-)" in l ]

    if isinstance(plus_labels, str) and isinstance(minus_labels, str):
        plus_labels = [plus_labels]
        minus_labels =[minus_labels]
    elif (isinstance(plus_labels, list) and
          isinstance(minus_labels, list)):
        if len(plus_labels) != len(minus_labels):
            check_ValueError(data_merged, plus_labels, minus_labels)
            return
    else:
        check_ValueError(data_merged, plus_labels, minus_labels)
        return

    for plus, minus in zip(plus_labels, minus_labels):
        if data_merged[plus].dtype != data_merged[minus].dtype:
            check_ValueError(data_merged, plus_labels, minus_labels)
            return

    result = data_merged.stack_anomalous(labels[0], labels[1])
    centrics = data_merged.label_centrics()["CENTRIC"]
    assert len(result.columns) == (len(data_merged.columns)-len(plus_labels))
    assert len(result) == (2*(~centrics).sum()) + centrics.sum()
    assert result.spacegroup.xhm() == data_merged.spacegroup.xhm()
    

def test_stack_anomalous_unmerged(data_unmerged):
    """
    Test DataSet.stack_anomalous() raises ValueError with unmerged data
    """
    with pytest.raises(ValueError):
        result = data_unmerged.stack_anomalous()


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
    assert len(result.columns) == (len(data_merged.columns)+len(newcolumns))
    assert in_asu(result.get_hkls(), result.spacegroup).all()
    assert len(result) == ((~centrics).sum()/2) + centrics.sum()
    assert result.spacegroup.xhm() == data_merged.spacegroup.xhm()

        
def test_unstack_anomalous_unmerged(data_unmerged):
    """
    Test DataSet.unstack_anomalous() raises ValueError with unmerged data
    """
    with pytest.raises(ValueError):
        result = data_unmerged.unstack_anomalous()


@pytest.mark.parametrize("rangeindexed", [True, False])
def test_roundtrip_merged(data_merged, rangeindexed):
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

    # DataSet.merge() operations seem to recast index dtypes
    assert_frame_equal(result, data, check_index_type=False)
