import numpy as np
from pandas.api.extensions import ExtensionDtype
from pandas.core.dtypes.dtypes import register_extension_dtype

from .base import NumpyExtensionArray

@register_extension_dtype
class MTZIntDtype(ExtensionDtype):
    """Dtype for generic MTZ integer data"""
    
    name = 'MTZInt'
    type = np.int32
    kind = 'i'
    na_value = np.nan
    mtztype = "I"
    
    @property
    def _is_numeric(self):
        return True
    
    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from "
                            "'{}'".format(cls, string))

    @classmethod
    def construct_array_type(cls):
        return MTZIntArray

class MTZIntArray(NumpyExtensionArray):
    """ExtensionArray for supporting MtzIntDtype"""
    
    _dtype = MTZIntDtype()
    _itemsize = 8
    ndim = 1
    can_hold_na = True
    
    def __init__(self, values, copy=True, dtype=None):

        self.data = np.array(values, dtype='int32', copy=copy)
        if isinstance(dtype, str):
            MTZIntDtype.construct_array_type(dtype)
        elif dtype:
            assert isinstance(dtype, MTZIntDtype)

MTZIntArray._add_arithmetic_ops()
MTZIntArray._add_comparison_ops()
