import numpy as np
from pandas.api.extensions import ExtensionDtype
from pandas.core.dtypes.dtypes import register_extension_dtype

from .base import NumpyExtensionArray

@register_extension_dtype
class StandardDeviationDtype(ExtensionDtype):
    """Dtype for standard deviation of observables: J, F, D or other"""
    
    name = 'Stddev'
    type = np.float32
    kind = 'f'
    na_value = np.nan
    mtztype = "Q"
    
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
        return StandardDeviationArray

class StandardDeviationArray(NumpyExtensionArray):
    """ExtensionArray for supporting StandardDeviationDtype"""    
    _dtype = StandardDeviationDtype()
    pass

StandardDeviationArray._add_arithmetic_ops()
StandardDeviationArray._add_comparison_ops()

@register_extension_dtype
class StandardDeviationSFFriedelDtype(ExtensionDtype):
    """Dtype for standard deviation of F(+) or F(-)"""
    
    name = 'StddevSFFriedel'
    type = np.float32
    kind = 'f'
    na_value = np.nan
    mtztype = "L"
    
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
        return StandardDeviationSFFriedelArray

class StandardDeviationSFFriedelArray(NumpyExtensionArray):
    """ExtensionArray for supporting StandardDeviationSFFriedelDtype"""
    
    _dtype = StandardDeviationSFFriedelDtype()
    
    def __init__(self, values, copy=True, dtype=None):

        self.data = np.array(values, dtype='float32', copy=copy)
        if isinstance(dtype, str):
            StandardDeviationSFFriedelDtype.construct_array_type(dtype)
        elif dtype:
            assert isinstance(dtype, StandardDeviationSFFriedelDtype)

StandardDeviationSFFriedelArray._add_arithmetic_ops()
StandardDeviationSFFriedelArray._add_comparison_ops()

@register_extension_dtype
class StandardDeviationIFriedelDtype(ExtensionDtype):
    """Dtype for standard deviation of I(+) or I(-)"""
    
    name = 'StddevIFriedel'
    type = np.float32
    kind = 'f'
    na_value = np.nan
    mtztype = "M"
    
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
        return StandardDeviationIFriedelArray

class StandardDeviationIFriedelArray(NumpyExtensionArray):
    """ExtensionArray for supporting StandardDeviationIFriedelDtype"""
    
    _dtype = StandardDeviationIFriedelDtype()
    
    def __init__(self, values, copy=True, dtype=None):

        self.data = np.array(values, dtype='float32', copy=copy)
        if isinstance(dtype, str):
            StandardDeviationIFriedelDtype.construct_array_type(dtype)
        elif dtype:
            assert isinstance(dtype, StandardDeviationIFriedelDtype)

StandardDeviationIFriedelArray._add_arithmetic_ops()
StandardDeviationIFriedelArray._add_comparison_ops()
