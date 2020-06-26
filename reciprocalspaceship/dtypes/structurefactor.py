from pandas.core.dtypes.dtypes import register_extension_dtype
from .base import NumpyExtensionArray, NumpyFloat32ExtensionDtype

@register_extension_dtype
class StructureFactorAmplitudeDtype(NumpyFloat32ExtensionDtype):
    """Dtype for structure factor amplitude  data"""
    name = 'SFAmplitude'
    mtztype = "F"

    @classmethod
    def construct_array_type(cls):
        return StructureFactorAmplitudeArray

class StructureFactorAmplitudeArray(NumpyExtensionArray):
    """ExtensionArray for supporting StructureFactorAmplitudeDtype"""
    _dtype = StructureFactorAmplitudeDtype()
    pass

@register_extension_dtype
class FriedelStructureFactorAmplitudeDtype(NumpyFloat32ExtensionDtype):
    """
    Dtype for structure factor amplitude data from Friedel pairs -- 
    F(+) or F(-)
    """
    name = 'FriedelSFAmplitude'
    mtztype = "G"

    @classmethod
    def construct_array_type(cls):
        return FriedelStructureFactorAmplitudeArray

class FriedelStructureFactorAmplitudeArray(NumpyExtensionArray):
    """ExtensionArray for supporting FriedelStructureFactorAmplitudeDtype"""
    _dtype = FriedelStructureFactorAmplitudeDtype()
    pass

@register_extension_dtype
class NormalizedStructureFactorAmplitudeDtype(NumpyFloat32ExtensionDtype):
    """Dtype for normalized structure factor amplitude data"""
    name = 'NormalizedSFAmplitude'
    mtztype = "E"

    @classmethod
    def construct_array_type(cls):
        return NormalizedStructureFactorAmplitudeArray

class NormalizedStructureFactorAmplitudeArray(NumpyExtensionArray):
    """ExtensionArray for supporting NormalizedStructureFactorAmplitudeDtype"""
    _dtype = NormalizedStructureFactorAmplitudeDtype()
    pass
