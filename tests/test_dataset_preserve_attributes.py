import pytest
import reciprocalspaceship as rs
import gemmi

@pytest.mark.parametrize("check_isomorphous", [True, False])
@pytest.mark.parametrize("sg", [gemmi.SpaceGroup(19), gemmi.SpaceGroup(96)])
@pytest.mark.parametrize("ignore_index", [True, False])
def test_concat(data_fmodel, check_isomorphous, sg, ignore_index):
    """
    Test whether attributes of DataSet are preserved through calls to 
    pd.concat()
    """
    other = data_fmodel.copy(deep=True)
    other.spacegroup = sg
    if check_isomorphous and sg.number == 19:
        with pytest.raises(ValueError):
            result = rs.concat([data_fmodel, other], ignore_index=ignore_index,
                               check_isomorphous=check_isomorphous)
    else:
        result = rs.concat([data_fmodel, other], ignore_index=ignore_index,
                           check_isomorphous=check_isomorphous)
        assert isinstance(result, rs.DataSet)
        assert len(result) == len(data_fmodel)*2
        if ignore_index:
            assert result._cache_index_dtypes == {}
        for attr in data_fmodel._metadata:
            if attr ==  "_cache_index_dtypes":
                continue
            assert result.__getattr__(attr) == data_fmodel.__getattr__(attr)

@pytest.mark.parametrize("check_isomorphous", [True, False])
@pytest.mark.parametrize("sg", [gemmi.SpaceGroup(19), gemmi.SpaceGroup(96)])
@pytest.mark.parametrize("ignore_index", [True, False])
def test_append(data_fmodel, check_isomorphous, sg, ignore_index):
    """
    Test whether attributes of DataSet are preserved through calls to 
    DataSet.append()
    """
    other = data_fmodel.copy(deep=True)
    other.spacegroup = sg
    if check_isomorphous and sg.number == 19:
        with pytest.raises(ValueError):
            result = data_fmodel.append(other, ignore_index, check_isomorphous=check_isomorphous)
    else:
        result = data_fmodel.append(other, ignore_index, check_isomorphous=check_isomorphous)
        assert isinstance(result, rs.DataSet)
        assert len(result) == len(data_fmodel)*2
        if ignore_index:
            assert result._cache_index_dtypes == {}
        for attr in data_fmodel._metadata:
            if attr ==  "_cache_index_dtypes":
                continue
            assert result.__getattr__(attr) == data_fmodel.__getattr__(attr)
            
@pytest.mark.parametrize("check_isomorphous", [True, False])
@pytest.mark.parametrize("sg", [gemmi.SpaceGroup(19), gemmi.SpaceGroup(96)])
def test_merge(data_fmodel, check_isomorphous, sg):
    """
    Test whether attributes of DataSet are preserved through calls to 
    DataSet.merge()
    """
    right = data_fmodel.copy(deep=True)
    right.spacegroup = sg
    if check_isomorphous and sg.number == 19:
        with pytest.raises(ValueError):
            result = data_fmodel.merge(right, left_index=True, right_index=True,
                                       check_isomorphous=check_isomorphous)
    else:
        result = data_fmodel.merge(right, left_index=True, right_index=True,
                                   check_isomorphous=check_isomorphous)
        assert isinstance(result, rs.DataSet)
        assert len(result) == len(data_fmodel)
        assert len(result.columns) == len(data_fmodel.columns)*2
        for attr in data_fmodel._metadata:
            assert result.__getattr__(attr) == data_fmodel.__getattr__(attr)

@pytest.mark.parametrize("check_isomorphous", [True, False])
@pytest.mark.parametrize("sg", [gemmi.SpaceGroup(19), gemmi.SpaceGroup(96)])
def test_join(data_fmodel, check_isomorphous, sg):
    """
    Test whether attributes of DataSet are preserved through calls to 
    DataSet.join()
    """
    other = data_fmodel.copy(deep=True)
    other.spacegroup = sg
    if check_isomorphous and sg.number == 19:
        with pytest.raises(ValueError):
            result = data_fmodel.join(other, lsuffix="x", rsuffix="y", 
                                      check_isomorphous=check_isomorphous)
    else:
        result = data_fmodel.join(other, lsuffix="x", rsuffix="y",
                                  check_isomorphous=check_isomorphous)
        assert isinstance(result, rs.DataSet)
        assert len(result) == len(data_fmodel)
        assert len(result.columns) == len(data_fmodel.columns)*2
        for attr in data_fmodel._metadata:
            assert result.__getattr__(attr) == data_fmodel.__getattr__(attr)

