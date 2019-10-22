import numpy as np
from pandas.api.extensions import ExtensionDtype
from pandas.core.dtypes.dtypes import register_extension_dtype

from .base import NumpyExtensionArray

@register_extension_dtype
class BatchDtype(ExtensionDtype):
    """Dtype for representing batch numbers"""
    
    name = 'Batch'
    type = np.float32
    kind = 'f'
    na_value = np.nan
    mtztype = "B"
    
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
        return BatchArray

class BatchArray(NumpyExtensionArray):
    """ExtensionArray for supporting BatchDtype"""
    
    _dtype = BatchDtype()
    
    def __init__(self, values, copy=True, dtype=None):

        self.data = np.array(values, dtype='float32', copy=copy)
        if isinstance(dtype, str):
            BatchDtype.construct_array_type(dtype)
        elif dtype:
            assert isinstance(dtype, BatchDtype)

BatchArray._add_arithmetic_ops()
BatchArray._add_comparison_ops()
