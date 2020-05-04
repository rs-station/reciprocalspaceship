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

@register_extension_dtype
class HendricksonLattmanDtype(NumpyFloat32ExtensionDtype):
    """
    Dtype for representing phase probability coefficients 
    (Hendrickson-Lattman) in reflection tables
    """
    name = 'HendricksonLattman'
    mtztype = "A"

    @classmethod
    def construct_array_type(cls):
        return HendricksonLattmanArray

class HendricksonLattmanArray(NumpyExtensionArray):
    """ExtensionArray for supporting HendricksonLattmanDtype"""
    _dtype = HendricksonLattmanDtype()
    pass
