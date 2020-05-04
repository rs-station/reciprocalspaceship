from pandas.core.dtypes.dtypes import register_extension_dtype
from .base import NumpyExtensionArray, NumpyFloat32ExtensionDtype

@register_extension_dtype
class AnomalousDifferenceDtype(NumpyFloat32ExtensionDtype):
    """Dtype for anomalous difference data in reflection tables"""
    name = 'AnomalousDifference'
    mtztype = "D"
    
    @classmethod
    def construct_array_type(cls):
        return AnomalousDifferenceArray

class AnomalousDifferenceArray(NumpyExtensionArray):
    """ExtensionArray for supporting AnomalousDifferenceDtype"""
    _dtype = AnomalousDifferenceDtype()
    pass
