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
class IntensityFriedelDtype(NumpyFloat32ExtensionDtype):
    """Dtype for I(+) or I(-) data in reflection tables"""
    name = 'IntensityFriedel'
    mtztype = "K"

    @classmethod
    def construct_array_type(cls):
        return IntensityFriedelArray

class IntensityFriedelArray(NumpyExtensionArray):
    """ExtensionArray for supporting IntensityFriedelDtype"""    
    _dtype = IntensityFriedelDtype()
    pass
