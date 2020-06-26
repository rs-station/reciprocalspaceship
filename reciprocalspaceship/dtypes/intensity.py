from pandas.core.dtypes.dtypes import register_extension_dtype
from .base import NumpyExtensionArray, NumpyFloat32ExtensionDtype

@register_extension_dtype
class IntensityDtype(NumpyFloat32ExtensionDtype):
    """Dtype for Intensity data in reflection tables"""
    name = 'Intensity'
    mtztype = "J"

    @classmethod
    def construct_array_type(cls):
        return IntensityArray

class IntensityArray(NumpyExtensionArray):
    """ExtensionArray for supporting IntensityDtype"""    
    _dtype = IntensityDtype()
    pass

@register_extension_dtype
class FriedelIntensityDtype(NumpyFloat32ExtensionDtype):
    """Dtype for I(+) or I(-) data in reflection tables"""
    name = 'FriedelIntensity'
    mtztype = "K"

    @classmethod
    def construct_array_type(cls):
        return FriedelIntensityArray

class FriedelIntensityArray(NumpyExtensionArray):
    """ExtensionArray for supporting FriedelIntensityDtype"""
    _dtype = FriedelIntensityDtype()
    pass
