import pytest
import unittest
import numpy as np
import pandas as pd
import reciprocalspaceship as rs
from pandas.tests.extension import base

array = {
    "HKL": rs.dtypes.hklindex.HKLIndexArray,
    "MTZInt": rs.dtypes.mtzint.MTZIntArray,
    "Batch": rs.dtypes.batch.BatchArray,
    "M_Isym": rs.dtypes.m_isym.M_IsymArray
}

@pytest.fixture(
    params=[
        rs.HKLIndexDtype,
        rs.MTZIntDtype,
        rs.BatchDtype,
        rs.M_IsymDtype
    ]
)
def dtype(request):
    return request.param()

@pytest.fixture
def data(dtype):
    return array[dtype.name]._from_sequence(np.arange(0, 100), dtype=dtype)

@pytest.fixture
def data_for_twos(dtype):
    return array[dtype.name]._from_sequence(np.ones(100) * 2, dtype=dtype)

@pytest.fixture
def data_missing(dtype):
    return array[dtype.name]._from_sequence([np.nan, 1.], dtype=dtype)

@pytest.fixture
def data_for_sorting(dtype):
    return array[dtype.name]._from_sequence([1., 2., 0.], dtype=dtype)

@pytest.fixture
def data_missing_for_sorting(dtype):
    return array[dtype.name]._from_sequence([1., np.nan, 0.], dtype=dtype)

@pytest.fixture(params=['data', 'data_missing'])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == 'data':
        return data
    elif request.param == 'data_missing':
        return data_missing
    
@pytest.fixture
def na_value(dtype):
    return dtype.na_value

@pytest.fixture
def na_cmp():
    # we are pd.nan
    return lambda x, y: pd.isna(x) and pd.isna(y)

@pytest.fixture
def data_for_grouping(dtype):
    b = 1
    a = 0
    c = 2
    na = np.nan
    return array[dtype.name]._from_sequence([b, b, na, na, a, a, b, c], dtype=dtype)

class TestCasting(base.BaseCastingTests):
    pass

class TestConstructors(base.BaseConstructorsTests):
    pass

class TestDtype(base.BaseDtypeTests):
    pass

class TestGetitem(base.BaseGetitemTests):
    pass

class TestGroupby(base.BaseGroupbyTests):
    pass

class TestInterface(base.BaseInterfaceTests):
    pass

class TestIO(base.BaseParsingTests):
    pass

class TestMethods(base.BaseMethodsTests):

    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna):
        all_data = all_data[:10]
        if dropna:
            other = all_data[~all_data.isna()]
        else:
            other = all_data

        result = rs.CrystalSeries(all_data).value_counts(dropna=dropna).sort_index()
        expected = rs.CrystalSeries(other).value_counts(dropna=dropna).sort_index()

        self.assert_series_equal(result, expected)
    pass

class TestMissing(base.BaseMissingTests):
    pass

class TestPrinting(base.BasePrintingTests):
    pass

class TestReshaping(base.BaseReshapingTests):
    pass

class TestSetitem(base.BaseSetitemTests):
    pass
