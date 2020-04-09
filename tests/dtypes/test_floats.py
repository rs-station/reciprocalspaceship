"""
Pandas unittests for ExtensionDtypes and ExtensionArrays for custom 
reciprocalspaceship dtypes that are backed by numpy float32 arrays.
"""

import pytest
import unittest
import numpy as np
import reciprocalspaceship as rs
import pandas as pd
from pandas.tests.extension import base

array = {
    "Intensity": rs.dtypes.intensity.IntensityArray,
    "SFAmplitude": rs.dtypes.structurefactor.StructureFactorAmplitudeArray,
    "AnomalousDifference": rs.dtypes.anomalousdifference.AnomalousDifferenceArray,
    "Stddev": rs.dtypes.stddev.StandardDeviationArray,
    "SFAmplitudeFriedel": rs.dtypes.structurefactor.StructureFactorAmplitudeFriedelArray,
    "StddevSFFriedel": rs.dtypes.stddev.StandardDeviationSFFriedelArray,
    "IntensityFriedel": rs.dtypes.intensity.IntensityFriedelArray,
    "StddevIFriedel": rs.dtypes.stddev.StandardDeviationIFriedelArray,
    "F_over_eps": rs.dtypes.structurefactor.ScaledStructureFactorAmplitudeArray,
    "Phase": rs.dtypes.phase.PhaseArray,
    "Weight": rs.dtypes.weight.WeightArray,
    "HendricksonLattman": rs.dtypes.phase.HendricksonLattmanArray,
    "MTZReal": rs.dtypes.mtzreal.MTZRealArray
}

@pytest.fixture(
    params=[
        rs.IntensityDtype,
        rs.StructureFactorAmplitudeDtype,
        rs.AnomalousDifferenceDtype,
        rs.StandardDeviationDtype,
        rs.StructureFactorAmplitudeFriedelDtype,
        rs.StandardDeviationSFFriedelDtype,
        rs.IntensityFriedelDtype,
        rs.StandardDeviationIFriedelDtype,
        rs.ScaledStructureFactorAmplitudeDtype,
        rs.PhaseDtype,        
        rs.WeightDtype,
        rs.HendricksonLattmanDtype,
        rs.MTZRealDtype
    ]
)
def dtype(request):
    return request.param()

@pytest.fixture
def data(dtype):
    return array[dtype.name](np.arange(0, 100), dtype=dtype)

@pytest.fixture
def data_for_twos(dtype):
    return array[dtype.name](np.ones(100) * 2, dtype=dtype)

@pytest.fixture
def data_missing(dtype):
    return array[dtype.name]([np.nan, 1.], dtype=dtype)

@pytest.fixture
def data_for_sorting(dtype):
    return array[dtype.name]([1., 2., 0.], dtype=dtype)

@pytest.fixture
def data_missing_for_sorting(dtype):
    return array[dtype.name]([1., np.nan, 0.], dtype=dtype)

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
    return lambda x, y: pd.isna(x) and pd.isna(y)

@pytest.fixture
def data_for_grouping(dtype):
    b = 1
    a = 0
    c = 2
    na = np.nan
    return array[dtype.name]([b, b, na, na, a, a, b, c], dtype=dtype)

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

        result = rs.DataSeries(all_data).value_counts(dropna=dropna).sort_index()
        expected = rs.DataSeries(other).value_counts(dropna=dropna).sort_index()

        print(result)
        print(expected)
        
        self.assert_series_equal(result, expected)

    @pytest.mark.xfail(reason="seems to be due to use of ExtensionArray")
    def test_combine_le(self, data_repeated):
        # GH 20825
        # Test that combine works when doing a <= (le) comparison
        orig_data1, orig_data2 = data_repeated(2)
        s1 = rs.DataSeries(orig_data1)
        s2 = rs.DataSeries(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 <= x2)
        expected = rs.DataSeries(
            [a <= b for (a, b) in zip(list(orig_data1), list(orig_data2))]
        )
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

        # Since our custom dtype defaults to a float32, the pd.concat()
        # call should upcast to a float64 when joining a float32 and int64
        result = pd.concat([df1["A"], df2["A"]])
        expected = pd.concat([df1["A"].astype("float64"), df2["A"].astype("float64")])
        self.assert_series_equal(result, expected)

class TestSetitem(base.BaseSetitemTests):
    pass
