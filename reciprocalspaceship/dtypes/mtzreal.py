from pandas.core.dtypes.dtypes import register_extension_dtype
from .base import NumpyExtensionArray, NumpyFloat32ExtensionDtype

@register_extension_dtype
class MTZRealDtype(NumpyFloat32ExtensionDtype):
    """Dtype for generic MTZ real data"""
    
    name = 'MTZReal'
    mtztype = "R"

    @classmethod
    def construct_array_type(cls):
        return MTZRealArray

class MTZRealArray(NumpyExtensionArray):
    """ExtensionArray for supporting MtzRealDtype"""
    _dtype = MTZRealDtype()
    pass
