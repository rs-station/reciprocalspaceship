import pytest
import numpy as np
import pandas as pd
import  reciprocalspaceship as rs

@pytest.mark.parametrize("level", [None, ["H", "K", "L"], ["H"]])
@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
def test_reset_index(data_fmodel, level, drop, inplace):
    """Test DataSet.reset_index()"""
    result = data_fmodel.reset_index(level=level, drop=drop, inplace=inplace)

    # Determine new index names
    if level is None:
        index_names = [None]
        columns = ["H", "K", "L"]
        cache = []
    elif level == ["H", "K", "L"]:
        index_names = [None]
        columns = ["H", "K", "L"]
        cache = []
    elif level == ["H"]:
        index_names = ["K", "L"]
        columns = ["H"]
        cache = ["K", "L"]
        
    if inplace:
        assert result is None
        assert data_fmodel.index.names == index_names
        for c in columns:
            if drop:
                assert c not in data_fmodel.columns
            else:
                assert c in data_fmodel.columns
        assert cache == list(data_fmodel._index_dtypes.keys())
        
    else:
        assert id(result) != id(data_fmodel)
        assert result.index.names == index_names
        assert data_fmodel.index.names != index_names
        for c in columns:
            if drop:
                assert c not in result.columns
                assert c not in data_fmodel.columns
            else:
                assert c in result.columns
                assert c not in data_fmodel.columns
        assert cache == list(result._index_dtypes.keys())
        assert cache != list(data_fmodel._index_dtypes.keys())


@pytest.mark.parametrize("keys", [
    ["H", "K", "L"],
    ["H"],
    "H",
    np.arange(168),
    [np.arange(168)],
    ["H", np.arange(168), "K"],
    rs.DataSeries(np.arange(168), name="temp"),
    [rs.DataSeries(np.arange(168), name="temp", dtype="I"), rs.DataSeries(np.arange(168), name="temp2", dtype="I")]
])
def test_set_index_cache(data_fmodel, keys):
    """
    Test DataSet.set_index() correctly sets DataSet._index_dtypes attribute

    Note
    ----
    There are 168 rows in data_fmodel
    """
    temp = data_fmodel.reset_index()
    result = temp.set_index(keys)

    if not isinstance(keys, list):
        keys = [keys]

    expected = sum([1 for k in keys if isinstance(k, (str, pd.Index, pd.Series))])
    assert len(result._index_dtypes) == expected

    for key in keys:
        if isinstance(key, str):
            assert result._index_dtypes[key] == temp[key].dtype.name
        elif isinstance(key, pd.Series):
            assert result._index_dtypes[key.name] == key.dtype.name
        elif isinstance(key, np.ndarray):
            assert None not in result._index_dtypes
