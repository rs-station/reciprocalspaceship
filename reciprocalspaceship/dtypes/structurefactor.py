import numpy as np
from pandas.api.extensions import ExtensionDtype
from pandas.core.dtypes.dtypes import register_extension_dtype

from .base import NumpyExtensionArray

@register_extension_dtype
class StructureFactorAmplitudeDtype(ExtensionDtype):
    """Dtype for structure factor amplitude  data"""
    
    name = 'StructureFactorAmplitude'
    type = np.float32
    kind = 'f'
    na_value = np.nan
    mtztype = "F"
    
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
        return StructureFactorAmplitudeArray

class StructureFactorAmplitudeArray(NumpyExtensionArray):
    """ExtensionArray for supporting StructureFactorAmplitudeDtype"""
    
    _dtype = StructureFactorAmplitudeDtype()
    
    def __init__(self, values, copy=True, dtype=None):

        self.data = np.array(values, dtype='float32', copy=copy)
        if isinstance(dtype, str):
            StructureFactorAmplitudeDtype.construct_array_type(dtype)
        elif dtype:
            assert isinstance(dtype, StructureFactorAmplitudeDtype)

StructureFactorAmplitudeArray._add_arithmetic_ops()
StructureFactorAmplitudeArray._add_comparison_ops()
