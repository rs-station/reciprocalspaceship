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

StructureFactorAmplitudeArray._add_arithmetic_ops()
StructureFactorAmplitudeArray._add_comparison_ops()

@register_extension_dtype
class StructureFactorAmplitudeFriedelDtype(NumpyFloat32ExtensionDtype):
    """
    Dtype for structure factor amplitude data from Friedel pairs -- 
    F(+) or F(-)
    """
    name = 'SFAmplitudeFriedel'
    mtztype = "G"

    @classmethod
    def construct_array_type(cls):
        return StructureFactorAmplitudeFriedelArray

class StructureFactorAmplitudeFriedelArray(NumpyExtensionArray):
    """ExtensionArray for supporting StructureFactorAmplitudeFriedelDtype"""
    _dtype = StructureFactorAmplitudeFriedelDtype()
    pass

StructureFactorAmplitudeFriedelArray._add_arithmetic_ops()
StructureFactorAmplitudeFriedelArray._add_comparison_ops()

@register_extension_dtype
class ScaledStructureFactorAmplitudeDtype(NumpyFloat32ExtensionDtype):
    """Dtype for structure factor amplitude  data"""
    name = 'F_over_eps'
    mtztype = "E"

    @classmethod
    def construct_array_type(cls):
        return ScaledStructureFactorAmplitudeArray

class ScaledStructureFactorAmplitudeArray(NumpyExtensionArray):
    """ExtensionArray for supporting ScaledStructureFactorAmplitudeDtype"""
    _dtype = ScaledStructureFactorAmplitudeDtype()
    pass

ScaledStructureFactorAmplitudeArray._add_arithmetic_ops()
ScaledStructureFactorAmplitudeArray._add_comparison_ops()
