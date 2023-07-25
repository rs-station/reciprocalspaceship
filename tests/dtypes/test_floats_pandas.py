"""
Pandas unittests for ExtensionDtypes and ExtensionArrays for custom
reciprocalspaceship dtypes that are backed by numpy float32 arrays.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.tests.extension import base
from pandas.tests.extension.test_floating import (
    TestComparisonOps as FloatingTestComparisonOps,
)

import reciprocalspaceship as rs


@pytest.fixture(
    params=[
        rs.IntensityDtype,
        rs.StructureFactorAmplitudeDtype,
        rs.AnomalousDifferenceDtype,
        rs.StandardDeviationDtype,
        rs.FriedelStructureFactorAmplitudeDtype,
        rs.StandardDeviationFriedelSFDtype,
        rs.FriedelIntensityDtype,
        rs.StandardDeviationFriedelIDtype,
        rs.NormalizedStructureFactorAmplitudeDtype,
        rs.PhaseDtype,
        rs.WeightDtype,
        rs.HendricksonLattmanDtype,
        rs.MTZRealDtype,
    ]
)
def dtype(request):
    return request.param()


@pytest.fixture
def data(dtype):
    return pd.array(np.arange(0, 100), dtype=dtype)


@pytest.fixture
def data_for_twos(dtype):
    return pd.array(np.ones(100) * 2, dtype=dtype)


@pytest.fixture
def data_missing(dtype):
    return pd.array([np.nan, 1.0], dtype=dtype)


@pytest.fixture
def data_for_sorting(dtype):
    return pd.array([1.0, 2.0, 0.0], dtype=dtype)


@pytest.fixture
def data_missing_for_sorting(dtype):
    return pd.array([1.0, np.nan, 0.0], dtype=dtype)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_for_grouping(dtype):
    b = 1
    a = 0
    c = 2
    na = np.nan
    return pd.array([b, b, na, na, a, a, b, c], dtype=dtype)


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
        """
        Rewrite original test to use rs.DataSeries instead of pd.Series
        """
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

    def test_value_counts_with_normalize(self, data):
        # GH 33172
        data = data[:10].unique()
        values = np.array(data[~data.isna()])

        result = (
            rs.DataSeries(data, dtype=data.dtype)
            .value_counts(normalize=True)
            .sort_index()
        )

        expected = rs.DataSeries(
            [1 / len(values)] * len(values), index=result.index, name=result.name
        )
        self.assert_series_equal(result, expected)


class TestComparisonOps(FloatingTestComparisonOps):
    pass


class TestMissing(base.BaseMissingTests):
    pass


class TestBooleanReduce(base.BaseBooleanReduceTests):
    pass


class TestNumericReduce(base.BaseNumericReduceTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


class TestReshaping(base.BaseReshapingTests):
    pass


class TestSetitem(base.BaseSetitemTests):
    pass
