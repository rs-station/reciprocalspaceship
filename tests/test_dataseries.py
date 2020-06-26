import pytest
import numpy as np
import reciprocalspaceship as rs

@pytest.mark.parametrize("data", [None, [], np.linspace(-180, 180, 181)])
@pytest.mark.parametrize("name", [None, "test"])
@pytest.mark.parametrize("dtype", [object, "float32", rs.PhaseDtype()])
def test_constructor(data, name, dtype):
    """Test constructor of DataSeries"""

    ds = rs.DataSeries(data, name=name, dtype=dtype)
    assert isinstance(ds, rs.DataSeries)
    assert ds.name == name
    if data is None:
        assert len(ds) == 0
        assert np.array_equal(
            ds.to_numpy(dtype=float),
            np.array([], dtype=float)
        )
    else:
        assert len(ds) == len(data)
        assert np.array_equal(
            ds.to_numpy(dtype=float),
            np.array(data, dtype=float)
        )

@pytest.mark.parametrize("data", [None, [], np.linspace(-180, 180, 181)])
@pytest.mark.parametrize("series_name", [None, "series_name"])
@pytest.mark.parametrize("frame_name", [None, "frame_name"])
@pytest.mark.parametrize("dtype", [object, "float32", rs.PhaseDtype()])
def test_constructor_expanddim(data, series_name, frame_name, dtype):
    """Test DataSeries.to_frame()"""

    ds = rs.DataSeries(data, name=series_name, dtype=dtype)
    d = ds.to_frame(name=frame_name)
    assert isinstance(d, rs.DataSet)
    assert len(d.columns) == 1
    assert isinstance(d.dtypes[0], type(ds.dtype))
    
    # Test hierarchy for column naming
    if frame_name:
        assert d.columns[0]  == frame_name
    elif series_name:
        assert d.columns[0] == series_name
    else:
        assert d.columns[0] == 0

@pytest.mark.parametrize("name", ["Name", "SIGF", "SIGI"])
def test_to_friedel_dtype(dtype_all, name):
    """Test DataSeries.to_friedel_dtype"""
    ds = rs.DataSeries(np.arange(0, 10), name=name, dtype=dtype_all[0]())
    result = ds.to_friedel_dtype()
    expected = dtype_all[0]
    if isinstance(ds.dtype, rs.StandardDeviationDtype):
        if name == "SIGF":
            expected = rs.StandardDeviationFriedelSFDtype
        elif name == "SIGI":
            expected = rs.StandardDeviationFriedelIDtype
    elif isinstance(ds.dtype, rs.IntensityDtype):
        expected = rs.FriedelIntensityDtype
    elif isinstance(ds.dtype, rs.StructureFactorAmplitudeDtype):
        expected = rs.FriedelStructureFactorAmplitudeDtype

    assert isinstance(result.dtype, expected)
    assert result.name == name
    assert np.array_equal(result.to_numpy(), np.arange(0, 10))

def test_from_friedel_dtype(dtype_all):
    """Test DataSeries.from_friedel_dtype"""
    ds = rs.DataSeries(np.arange(0, 10), dtype=dtype_all[0]())
    result = ds.from_friedel_dtype()
    expected = dtype_all[0]
    if (isinstance(ds.dtype, rs.StandardDeviationFriedelSFDtype) or
        isinstance(ds.dtype, rs.StandardDeviationFriedelIDtype)):
            expected = rs.StandardDeviationDtype
    elif isinstance(ds.dtype, rs.FriedelIntensityDtype):
        expected = rs.IntensityDtype
    elif isinstance(ds.dtype, rs.FriedelStructureFactorAmplitudeDtype):
        expected = rs.StructureFactorAmplitudeDtype

    assert isinstance(result.dtype, expected)
    assert np.array_equal(result.to_numpy(), np.arange(0, 10))










