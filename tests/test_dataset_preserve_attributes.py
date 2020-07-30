import pytest
import reciprocalspaceship as rs

def test_concat(data_fmodel):
    """
    Test whether attributes of DataSet are preserved through calls to 
    pd.concat()
    """
    result = rs.concat([data_fmodel, data_fmodel])
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

