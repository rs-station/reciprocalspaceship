import numpy as np
from pandas.api.extensions import ExtensionDtype
from pandas.core.dtypes.dtypes import register_extension_dtype

from .base import NumpyExtensionArray

@register_extension_dtype
class WeightDtype(ExtensionDtype):
    """Dtype for representing weights"""
    
    name = 'Weight'
    type = np.float32
    kind = 'f'
    na_value = np.nan
    mtztype = "W"
    
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
        return WeightArray

class WeightArray(NumpyExtensionArray):
    """ExtensionArray for supporting WeightDtype"""
    
    _dtype = WeightDtype()
    
    def __init__(self, values, copy=True, dtype=None):

        self.data = np.array(values, dtype='float32', copy=copy)
        if isinstance(dtype, str):
            WeightDtype.construct_array_type(dtype)
        elif dtype:
            assert isinstance(dtype, WeightDtype)

WeightArray._add_arithmetic_ops()
WeightArray._add_comparison_ops()
