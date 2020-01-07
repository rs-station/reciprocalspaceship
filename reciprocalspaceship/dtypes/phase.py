import numpy as np
from pandas.core.dtypes.dtypes import register_extension_dtype

from .base import NumpyExtensionArray, NumpyFloat32ExtensionDtype

@register_extension_dtype
class PhaseDtype(NumpyFloat32ExtensionDtype):
    """Dtype for representing phase data in reflection tables"""
    name = 'Phase'
    mtztype = "P"
    
    @classmethod
    def construct_array_type(cls):
        return PhaseArray
    
class PhaseArray(NumpyExtensionArray):
    """ExtensionArray for supporting PhaseDtype"""
    _dtype = PhaseDtype()
    pass

PhaseArray._add_arithmetic_ops()
PhaseArray._add_comparison_ops()

@register_extension_dtype
class HendricksonLattmanDtype(ExtensionDtype):
    """
    Dtype for representing phase probability coefficients 
    (Hendrickson-Lattman)
    """
    
    name = 'HendricksonLattman'
    type = np.float32
    kind = 'f'
    na_value = np.nan
    mtztype = "A"
    
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
        return HendricksonLattmanArray

class HendricksonLattmanArray(NumpyExtensionArray):
    """ExtensionArray for supporting HendricksonLattmanDtype"""
    _dtype = HendricksonLattmanDtype()
    pass

HendricksonLattmanArray._add_arithmetic_ops()
HendricksonLattmanArray._add_comparison_ops()
