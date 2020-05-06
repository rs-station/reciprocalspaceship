import pytest
import numpy as np
import reciprocalspaceship as rs

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

def test_numpy_dtype(dtype):
    # Test NumpyFloat32ExtensionDtype.numpy_dtype
    assert dtype.numpy_dtype == dtype.type

def test_itemsize(dtype):
    # Test NumpyFloat32ExtensionDtype.itemsize
    assert dtype.itemsize == dtype.type().itemsize
