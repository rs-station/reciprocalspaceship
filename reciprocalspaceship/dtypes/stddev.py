from pandas.core.dtypes.dtypes import register_extension_dtype
from .base import NumpyExtensionArray, NumpyFloat32ExtensionDtype

@register_extension_dtype
class StandardDeviationDtype(NumpyFloat32ExtensionDtype):
    """Dtype for standard deviation of observables: J, F, D or other"""
    name = 'Stddev'
    mtztype = "Q"
    
    @classmethod
    def construct_array_type(cls):
        return StandardDeviationArray

class StandardDeviationArray(NumpyExtensionArray):
    """ExtensionArray for supporting StandardDeviationDtype"""    
    _dtype = StandardDeviationDtype()
    pass

@register_extension_dtype
class StandardDeviationSFFriedelDtype(NumpyFloat32ExtensionDtype):
    """Dtype for standard deviation of F(+) or F(-)"""
    name = 'StddevSFFriedel'
    mtztype = "L"

    @classmethod
    def construct_array_type(cls):
        return StandardDeviationSFFriedelArray

class StandardDeviationSFFriedelArray(NumpyExtensionArray):
    """ExtensionArray for supporting StandardDeviationSFFriedelDtype"""
    _dtype = StandardDeviationSFFriedelDtype()
    pass

@register_extension_dtype
class StandardDeviationIFriedelDtype(NumpyFloat32ExtensionDtype):
    """Dtype for standard deviation of I(+) or I(-)"""
    name = 'StddevIFriedel'
    mtztype = "M"

    @classmethod
    def construct_array_type(cls):
        return StandardDeviationIFriedelArray

class StandardDeviationIFriedelArray(NumpyExtensionArray):
    """ExtensionArray for supporting StandardDeviationIFriedelDtype"""
    _dtype = StandardDeviationIFriedelDtype()
    pass
