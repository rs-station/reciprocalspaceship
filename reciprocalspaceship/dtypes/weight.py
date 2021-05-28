from pandas.core.dtypes.dtypes import register_extension_dtype
from .base import MTZFloatArray, MTZFloat32Dtype

@register_extension_dtype
class WeightDtype(MTZFloat32Dtype):
    """Dtype for representing weights"""
    name = 'Weight'
    mtztype = "W"
    
    @classmethod
    def construct_array_type(cls):
        return WeightArray

class WeightArray(MTZFloatArray):
    """ExtensionArray for supporting WeightDtype"""
    _dtype = WeightDtype()
    pass
