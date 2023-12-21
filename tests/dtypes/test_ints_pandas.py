import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest
from pandas.core.dtypes.common import is_float_dtype, is_signed_integer_dtype
from pandas.testing import assert_series_equal
from pandas.tests.extension import base

import reciprocalspaceship as rs


@pytest.fixture(
    params=[rs.HKLIndexDtype, rs.MTZIntDtype, rs.BatchDtype, rs.M_IsymDtype]
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


class TestIndex(base.BaseIndexTests):
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

        assert_series_equal(result, expected)

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
        assert_series_equal(result, expected)


class TestComparisonOps(base.BaseComparisonOpsTests):
    def _compare_other(self, ser: pd.Series, data, op, other):
        if op.__name__ in ["eq", "ne", "le", "ge", "lt", "gt"]:
            # comparison should match point-wise comparisons
            result = op(ser, other)
            expected = ser.combine(other, op).astype("boolean")
            assert_series_equal(result, expected)

        else:
            exc = None
            try:
                result = op(ser, other)
            except Exception as err:
                exc = err

            if exc is None:
                # Didn't error, then should match pointwise behavior
                expected = ser.combine(other, op)
                assert_series_equal(result, expected)
            else:
                with pytest.raises(type(exc)):
                    ser.combine(other, op)

    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        sdtype = tm.get_dtype(obj)
        expected = pointwise_result

        if op_name in ("eq", "ne", "le", "ge", "lt", "gt"):
            return expected.astype("boolean")

        if sdtype.kind in "iu":
            if op_name in ("__rtruediv__", "__truediv__", "__div__"):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        "Downcasting object dtype arrays",
                        category=FutureWarning,
                    )
                    filled = expected.fillna(np.nan)
                expected = filled.astype("Float64")
            else:
                # combine method result in 'biggest' (int64) dtype
                expected = expected.astype(sdtype)
        elif sdtype.kind == "b":
            if op_name in (
                "__floordiv__",
                "__rfloordiv__",
                "__pow__",
                "__rpow__",
                "__mod__",
                "__rmod__",
            ):
                # combine keeps boolean type
                expected = expected.astype("Int8")

            elif op_name in ("__truediv__", "__rtruediv__"):
                # combine with bools does not generate the correct result
                #  (numpy behaviour for div is to regard the bools as numeric)
                op = self.get_op_from_name(op_name)
                expected = self._combine(obj.astype(float), other, op)
                expected = expected.astype("Float64")

            if op_name == "__rpow__":
                # for rpow, combine does not propagate NaN
                result = getattr(obj, op_name)(other)
                expected[result.isna()] = np.nan
        else:
            # combine method result in 'biggest' (float64) dtype
            expected = expected.astype(sdtype)
        return expected


class TestMissing(base.BaseMissingTests):
    pass


class TestBooleanReduce(base.BaseBooleanReduceTests):
    pass


class TestNumericReduce(base.BaseNumericReduceTests):
    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        """
        Handle expected return types for reductions that may change int32-backed dtype
        """
        # Floats can stay the same
        if is_float_dtype(arr.dtype):
            cmp_dtype = arr.dtype.name
        # These reductions cannot always be safely cast to int32
        elif op_name in ["mean", "median", "var", "std", "skew"]:
            cmp_dtype = "Float64"
        else:
            cmp_dtype = arr.dtype.name
        return cmp_dtype


class TestParsing(base.BaseParsingTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


class TestReshaping(base.BaseReshapingTests):
    pass


class TestSetitem(base.BaseSetitemTests):
    pass
