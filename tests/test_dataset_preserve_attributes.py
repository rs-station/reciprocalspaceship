import pytest
import reciprocalspaceship as rs
import gemmi

@pytest.mark.parametrize("check_isomorphous", [True, False])
@pytest.mark.parametrize("sg", [gemmi.SpaceGroup(19), gemmi.SpaceGroup(96)])
def test_concat(data_fmodel, check_isomorphous, sg):
    """
    Test whether attributes of DataSet are preserved through calls to 
    pd.concat()
    """
    other = data_fmodel.copy(deep=True)
    other.spacegroup = sg
    if check_isomorphous and sg.number == 19:
        with pytest.raises(ValueError):
            result = rs.concat([data_fmodel, other], check_isomorphous=check_isomorphous)
    else:
        result = rs.concat([data_fmodel, other], check_isomorphous=check_isomorphous)
        assert isinstance(result, rs.DataSet)
        assert len(result) == len(data_fmodel)*2
        for attr in data_fmodel._metadata:
            assert result.__getattr__(attr) == data_fmodel.__getattr__(attr)


def test_append(data_fmodel):
    """
    Test whether attributes of DataSet are preserved through calls to 
    DataSet.append()
    """
    result = data_fmodel.append(data_fmodel)

    assert isinstance(result, rs.DataSet)
    assert len(result) == len(data_fmodel)*2
    for attr in data_fmodel._metadata:
        assert result.__getattr__(attr) == data_fmodel.__getattr__(attr)

