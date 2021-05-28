from pandas.core.dtypes.dtypes import register_extension_dtype
from .base import MTZFloatArray, MTZFloat32Dtype

@register_extension_dtype
class StandardDeviationDtype(MTZFloat32Dtype):
    """Dtype for standard deviation of observables: J, F, D or other"""
    name = 'Stddev'
    mtztype = "Q"
    
    @classmethod
    def construct_array_type(cls):
        return StandardDeviationArray

class StandardDeviationArray(MTZFloatArray):
    """ExtensionArray for supporting StandardDeviationDtype"""    
    _dtype = StandardDeviationDtype()
    pass

@register_extension_dtype
class StandardDeviationFriedelSFDtype(MTZFloat32Dtype):
    """Dtype for standard deviation of F(+) or F(-)"""
    name = 'StddevFriedelSF'
    mtztype = "L"

    @classmethod
    def construct_array_type(cls):
        return StandardDeviationFriedelSFArray

class StandardDeviationFriedelSFArray(MTZFloatArray):
    """ExtensionArray for supporting StandardDeviationFriedelSFDtype"""
    _dtype = StandardDeviationFriedelSFDtype()
    pass

@register_extension_dtype
class StandardDeviationFriedelIDtype(MTZFloat32Dtype):
    """Dtype for standard deviation of I(+) or I(-)"""
    name = 'StddevFriedelI'
    mtztype = "M"

    @classmethod
    def construct_array_type(cls):
        return StandardDeviationFriedelIArray

class StandardDeviationFriedelIArray(MTZFloatArray):
    """ExtensionArray for supporting StandardDeviationFriedelIDtype"""
    _dtype = StandardDeviationFriedelIDtype()
    pass
