import numpy as np
from pandas.api.extensions import ExtensionDtype
from pandas.core.dtypes.dtypes import register_extension_dtype

from .base import NumpyExtensionArray

@register_extension_dtype
class M_IsymDtype(ExtensionDtype):
    """Dtype for representing M/Isym values"""
    
    name = 'M_Isym'
    type = np.float32
    kind = 'f'
    na_value = np.nan
    mtztype = "Y"
    
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
        return M_IsymArray

class M_IsymArray(NumpyExtensionArray):
    """ExtensionArray for supporting M_IsymDtype"""
    
    _dtype = M_IsymDtype()
    
    def __init__(self, values, copy=True, dtype=None):

        self.data = np.array(values, dtype='float32', copy=copy)
        if isinstance(dtype, str):
            M_IsymDtype.construct_array_type(dtype)
        elif dtype:
            assert isinstance(dtype, M_IsymDtype)

M_IsymArray._add_arithmetic_ops()
M_IsymArray._add_comparison_ops()
