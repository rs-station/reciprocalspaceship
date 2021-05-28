from pandas.core.dtypes.dtypes import register_extension_dtype
from .base import MTZFloatArray, MTZFloat32Dtype

@register_extension_dtype
class IntensityDtype(MTZFloat32Dtype):
    """Dtype for Intensity data in reflection tables"""
    name = 'Intensity'
    mtztype = "J"

    @classmethod
    def construct_array_type(cls):
        return IntensityArray

class IntensityArray(MTZFloatArray):
    """ExtensionArray for supporting IntensityDtype"""    
    _dtype = IntensityDtype()
    pass

@register_extension_dtype
class FriedelIntensityDtype(MTZFloat32Dtype):
    """Dtype for I(+) or I(-) data in reflection tables"""
    name = 'FriedelIntensity'
    mtztype = "K"

    @classmethod
    def construct_array_type(cls):
        return FriedelIntensityArray

class FriedelIntensityArray(MTZFloatArray):
    """ExtensionArray for supporting FriedelIntensityDtype"""
    _dtype = FriedelIntensityDtype()
    pass
