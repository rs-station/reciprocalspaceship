import pytest
import numpy as np
import reciprocalspaceship as rs
from pandas.testing import assert_series_equal

def test_repr(dtype_ints):
    """Test MTZInt32Dtype.__repr__ returns dtype name"""
    assert dtype_ints[1] == dtype_ints[0]().__repr__()

def test_itemsize(dtype_floats):
    """Test NumpyFloat32ExtensionDtype.itemsize"""
    assert dtype_floats[0]().itemsize == dtype_floats[0]().type().itemsize

def test_numpy_dtype(dtype_floats):
    """Test NumpyFloat32ExtensionDtype.numpy_dtype"""
    assert dtype_floats[0]().numpy_dtype == dtype_floats[0]().type

def test_numpy_navalue(data_float):
    """Test NumpyExtensionArray.na_value returns np.nan"""
    assert data_float.na_value is np.nan

def test_numpy_tolist(data_float):
    """Test NumpyExtensionArray.tolist() returns list"""
    result = data_float.tolist()
    assert isinstance(result, list)
    assert np.array_equal(np.array(result), data_float.data)
    
def test_astype_singleletter(dtype_all):
    """Test DataSeries.astype() with single-letter mtztype"""
    expected = rs.DataSeries(np.arange(0, 100), dtype=dtype_all[0]())
    result = expected.astype(expected.dtype.mtztype)
    assert_series_equal(result, expected)

def test_astype_name(dtype_all):
    """Test DataSeries.astype() with name"""
    expected = rs.DataSeries(np.arange(0, 100), dtype=dtype_all[0]())
    result = expected.astype(expected.dtype.name)
    assert_series_equal(result, expected)
    assert expected.dtype.name == str(result.dtype)
