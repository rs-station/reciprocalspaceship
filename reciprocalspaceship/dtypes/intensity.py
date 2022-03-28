from pandas.core.dtypes.dtypes import register_extension_dtype

from reciprocalspaceship.dtypes.base import MTZFloat32Dtype, MTZFloatArray


@register_extension_dtype
class IntensityDtype(MTZFloat32Dtype):
    """Dtype for Intensity data in reflection tables"""

    name = "Intensity"
    mtztype = "J"

    def is_friedel_dtype(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return IntensityArray


class IntensityArray(MTZFloatArray):
    """ExtensionArray for supporting IntensityDtype"""

    _dtype = IntensityDtype()
    pass


@register_extension_dtype
class FriedelIntensityDtype(MTZFloat32Dtype):
    """Dtype for I(+) or I(-) data in reflection tables"""

    name = "FriedelIntensity"
    mtztype = "K"

    def is_friedel_dtype(self):
        return True

    @classmethod
    def construct_array_type(cls):
        return FriedelIntensityArray


class FriedelIntensityArray(MTZFloatArray):
    """ExtensionArray for supporting FriedelIntensityDtype"""

    _dtype = FriedelIntensityDtype()
    pass
