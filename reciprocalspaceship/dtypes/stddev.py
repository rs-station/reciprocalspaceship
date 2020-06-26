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
class StandardDeviationFriedelSFDtype(NumpyFloat32ExtensionDtype):
    """Dtype for standard deviation of F(+) or F(-)"""
    name = 'StddevFriedelSF'
    mtztype = "L"

    @classmethod
    def construct_array_type(cls):
        return StandardDeviationFriedelSFArray

class StandardDeviationFriedelSFArray(NumpyExtensionArray):
    """ExtensionArray for supporting StandardDeviationFriedelSFDtype"""
    _dtype = StandardDeviationFriedelSFDtype()
    pass

@register_extension_dtype
class StandardDeviationFriedelIDtype(NumpyFloat32ExtensionDtype):
    """Dtype for standard deviation of I(+) or I(-)"""
    name = 'StddevFriedelI'
    mtztype = "M"

    @classmethod
    def construct_array_type(cls):
        return StandardDeviationFriedelIArray

class StandardDeviationFriedelIArray(NumpyExtensionArray):
    """ExtensionArray for supporting StandardDeviationFriedelIDtype"""
    _dtype = StandardDeviationFriedelIDtype()
    pass
