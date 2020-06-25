import pytest
import numpy as np
import pandas as pd
import reciprocalspaceship as rs

@pytest.fixture
def na_value(dtype):
    return dtype.na_value

@pytest.fixture
def na_cmp():
    return lambda x, y: pd.isna(x) and pd.isna(y)

@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param

@pytest.fixture(
    params=[
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: rs.DataSeries([1] * len(x)),
        lambda x: x,
    ],
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(request):
    """
    Functions to test groupby.apply().
    """
    return request.param

@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param

@pytest.fixture(params=[True, False])
def as_array(request):
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
    return request.param

@pytest.fixture(params=[True, False])
def as_series(request):
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request):
    """
    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param

@pytest.fixture(params=[True, False])
def as_frame(request):
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param

@pytest.fixture
def data_repeated(data):
    """
    Generate many datasets.

    Parameters
    ----------
    data : fixture implementing `data`

    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """

    def gen(count):
        for _ in range(count):
            yield data

    return gen

array = {
    # Integer dtypes
    "HKL": rs.dtypes.hklindex.HKLIndexArray,
    "MTZInt": rs.dtypes.mtzint.MTZIntArray,
    "Batch": rs.dtypes.batch.BatchArray,
    "M_Isym": rs.dtypes.m_isym.M_IsymArray,

    # Float32 dtypes
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

integer_dtypes=[
    (rs.HKLIndexDtype, "HKL"),
    (rs.MTZIntDtype, "MTZInt"),
    (rs.BatchDtype, "Batch"),
    (rs.M_IsymDtype, "M_Isym")
]

float_dtypes=[
    (rs.IntensityDtype, "Intensity"),
    (rs.StructureFactorAmplitudeDtype, "SFAmplitude"),
    (rs.AnomalousDifferenceDtype, "AnomalousDifference"),
    (rs.StandardDeviationDtype, "Stddev"),
    (rs.StructureFactorAmplitudeFriedelDtype, "SFAmplitudeFriedel"),
    (rs.StandardDeviationSFFriedelDtype, "StddevSFFriedel"),
    (rs.IntensityFriedelDtype, "IntensityFriedel"),
    (rs.StandardDeviationIFriedelDtype, "StddevIFriedel"),
    (rs.ScaledStructureFactorAmplitudeDtype, "F_over_eps"),
    (rs.PhaseDtype, "Phase"),
    (rs.WeightDtype, "Weight"),
    (rs.HendricksonLattmanDtype, "HendricksonLattman"),
    (rs.MTZRealDtype, "MTZReal")
]
    
@pytest.fixture(params=integer_dtypes)
def dtype_ints(request):
    return request.param

@pytest.fixture(params=float_dtypes)
def dtype_floats(request):
    return request.param

@pytest.fixture(params=integer_dtypes + float_dtypes)
def dtype_all(request):
    return request.param

@pytest.fixture
def data_int(dtype_ints):
    return array[dtype_ints[0].name]._from_sequence(np.arange(0, 100),
                                                    dtype=dtype_ints[0]())

@pytest.fixture
def data_float(dtype_floats):
    return array[dtype_floats[0].name]._from_sequence(np.arange(0, 100),
                                                      dtype=dtype_floats[0]())

@pytest.fixture
def data_all(dtype_all):
    return array[dtype_all[0].name]._from_sequence(np.arange(0, 100),
                                                   dtype=dtype_all[0]())
