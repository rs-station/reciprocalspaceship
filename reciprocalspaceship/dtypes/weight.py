from pandas.core.dtypes.dtypes import register_extension_dtype
from .base import NumpyExtensionArray, NumpyFloat32ExtensionDtype

@register_extension_dtype
class WeightDtype(NumpyFloat32ExtensionDtype):
    """Dtype for representing weights"""
    name = 'Weight'
    mtztype = "W"
    
    @classmethod
    def construct_array_type(cls):
        return WeightArray

class WeightArray(NumpyExtensionArray):
    """ExtensionArray for supporting WeightDtype"""
    _dtype = WeightDtype()
    pass
