import numpy as np
from pandas.api.extensions import ExtensionDtype
from pandas.core.dtypes.dtypes import register_extension_dtype

from .base import NumpyExtensionArray

@register_extension_dtype
class IntensityDtype(ExtensionDtype):
    """Dtype for Intensity data"""
    
    name = 'Intensity'
    type = np.float32
    kind = 'f'
    na_value = np.nan
    mtztype = "J"
    
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
        return IntensityArray

class IntensityArray(NumpyExtensionArray):
    """ExtensionArray for supporting IntensityDtype"""
    
    _dtype = IntensityDtype()
    
    def __init__(self, values, copy=True, dtype=None):

        self.data = np.array(values, dtype='float32', copy=copy)
        if isinstance(dtype, str):
            IntensityDtype.construct_array_type(dtype)
        elif dtype:
            assert isinstance(dtype, IntensityDtype)

IntensityArray._add_arithmetic_ops()
IntensityArray._add_comparison_ops()

@register_extension_dtype
class IntensityFriedelDtype(ExtensionDtype):
    """Dtype for I(+) or I(-) data"""
    
    name = 'IntensityFriedel'
    type = np.float32
    kind = 'f'
    na_value = np.nan
    mtztype = "K"
    
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
        return IntensityFriedelArray

class IntensityFriedelArray(NumpyExtensionArray):
    """ExtensionArray for supporting IntensityFriedelDtype"""
    
    _dtype = IntensityFriedelDtype()
    
    def __init__(self, values, copy=True, dtype=None):

        self.data = np.array(values, dtype='float32', copy=copy)
        if isinstance(dtype, str):
            IntensityFriedelDtype.construct_array_type(dtype)
        elif dtype:
            assert isinstance(dtype, IntensityFriedelDtype)

IntensityFriedelArray._add_arithmetic_ops()
IntensityFriedelArray._add_comparison_ops()
