import pytest
import numpy as np
import reciprocalspaceship as rs

array = {
    "HKL": rs.dtypes.hklindex.HKLIndexArray,
    "MTZInt": rs.dtypes.mtzint.MTZIntArray,
    "Batch": rs.dtypes.batch.BatchArray,
    "M_Isym": rs.dtypes.m_isym.M_IsymArray
}

@pytest.fixture(
    params=[
        (rs.HKLIndexDtype, "HKL"),
        (rs.MTZIntDtype, "MTZInt"),
        (rs.BatchDtype, "Batch"),
        (rs.M_IsymDtype, "M_Isym")
    ]
)
def dtype(request):
    return request.param

@pytest.fixture
def data(dtype):
    return array[dtype[0].name]._from_sequence(np.arange(0, 100), dtype=dtype[0]())

def test_repr(dtype):
    # Test MTZInt32Dtype.__repr__ returns dtype name
    assert dtype[1] == str(dtype[0]())

def test_coerce_to_ndarray(data):
    # Test MTZInt32Dtype._coerce_to_ndarray defaults to returning
    # np.int32
    assert data._coerce_to_ndarray().dtype.type is np.int32
