import pytest

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
        assert cache == list(data_fmodel._cache_index_dtypes.keys())
        
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
        assert cache == list(result._cache_index_dtypes.keys())
        assert cache != list(data_fmodel._cache_index_dtypes.keys())
