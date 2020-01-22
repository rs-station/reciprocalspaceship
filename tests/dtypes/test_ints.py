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
    # we are np.nan
    return lambda x, y: np.isnan(x) and np.isnan(y)

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

        print(result)
        print(expected)
        
        self.assert_series_equal(result, expected)
        
class TestMissing(base.BaseMissingTests):
    pass

class TestPrinting(base.BasePrintingTests):
    pass

class TestReshaping(base.BaseReshapingTests):

    def test_concat_mixed_dtypes(self, data):
        # https://github.com/pandas-dev/pandas/issues/20762
        df1 = pd.DataFrame({"A": data[:3]})
        df2 = pd.DataFrame({"A": [1, 2, 3]})
        df3 = pd.DataFrame({"A": ["a", "b", "c"]}).astype("category")
        dfs = [df1, df2, df3]

        # dataframes
        result = pd.concat(dfs)
        expected = pd.concat([x.astype(object) for x in dfs])
        self.assert_frame_equal(result, expected)

        # series
        result = pd.concat([x["A"] for x in dfs])
        expected = pd.concat([x["A"].astype(object) for x in dfs])
        self.assert_series_equal(result, expected)

        # simple test for just EA and one other
        result = pd.concat([df1, df2])
        expected = pd.concat([df1.astype("object"), df2.astype("object")])
        self.assert_frame_equal(result, expected)

        # Since our custom dtype defaults to an int32, the pd.concat()
        # call should upcast to an int64 when joining an int32 and int64
        result = pd.concat([df1["A"], df2["A"]])
        expected = pd.concat([df1["A"].astype("int64"), df2["A"].astype("int64")])
        self.assert_series_equal(result, expected)

class TestSetitem(base.BaseSetitemTests):
    pass
