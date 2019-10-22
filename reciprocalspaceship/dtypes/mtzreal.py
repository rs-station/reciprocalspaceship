import numpy as np
from pandas.api.extensions import ExtensionDtype
from pandas.core.dtypes.dtypes import register_extension_dtype

from .base import NumpyExtensionArray

@register_extension_dtype
class MTZRealDtype(ExtensionDtype):
    """Dtype for generic MTZ real data"""
    
    name = 'MTZReal'
    type = np.float32
    kind = 'f'
    na_value = np.nan
    mtztype = "R"
    
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
        return MTZRealArray

class MTZRealArray(NumpyExtensionArray):
    """ExtensionArray for supporting MtzRealDtype"""
    
    _dtype = MTZRealDtype()
    _itemsize = 8
    ndim = 1
    can_hold_na = True
    
    def __init__(self, values, copy=True, dtype=None):

        self.data = np.array(values, dtype='float32', copy=copy)
        if isinstance(dtype, str):
            MTZRealDtype.construct_array_type(dtype)
        elif dtype:
            assert isinstance(dtype, MTZRealDtype)

MTZRealArray._add_arithmetic_ops()
MTZRealArray._add_comparison_ops()
