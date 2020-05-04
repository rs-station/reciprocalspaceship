import pytest
import numpy as np
import reciprocalspaceship as rs

@pytest.mark.parametrize("data", [None, [], np.linspace(-180, 180, 181)])
@pytest.mark.parametrize("name", [None, "test"])
@pytest.mark.parametrize("dtype", [object, "float32", rs.PhaseDtype()])
def test_constructor(data, name, dtype):
    # Test constructor of DataSeries

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
    # Test DataSeries.to_frame()

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
