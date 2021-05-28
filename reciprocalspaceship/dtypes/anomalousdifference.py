from pandas.core.dtypes.dtypes import register_extension_dtype
from .base import MTZFloatArray, MTZFloat32Dtype

@register_extension_dtype
class AnomalousDifferenceDtype(MTZFloat32Dtype):
    """Dtype for anomalous difference data in reflection tables"""
    name = 'AnomalousDifference'
    mtztype = "D"
    
    @classmethod
    def construct_array_type(cls):
        return AnomalousDifferenceArray

class AnomalousDifferenceArray(MTZFloatArray):
    """ExtensionArray for supporting AnomalousDifferenceDtype"""
    _dtype = AnomalousDifferenceDtype()
    pass
