import numpy as np
from pandas.api.extensions import ExtensionDtype
from pandas.core.dtypes.dtypes import register_extension_dtype

from .base import NumpyExtensionArray

@register_extension_dtype
class HKLIndexDtype(ExtensionDtype):
    """Dtype for HKL indices"""
    
    name = 'HKL'
    type = np.int32
    kind = 'i'
    na_value = np.nan
    mtztype = "H"
    
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
        return HKLIndexArray

class HKLIndexArray(NumpyExtensionArray):
    """ExtensionArray for supporting HKLIndexDtype"""
    
    _dtype = HKLIndexDtype()
    
    def __init__(self, values, copy=True, dtype=None):

        self.data = np.array(values, dtype='int32', copy=copy)
        if isinstance(dtype, str):
            HKLIndexDtype.construct_array_type(dtype)
        elif dtype:
            assert isinstance(dtype, HKLIndexDtype)

HKLIndexArray._add_arithmetic_ops()
HKLIndexArray._add_comparison_ops()
