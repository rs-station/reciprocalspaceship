from pandas.core.dtypes.dtypes import register_extension_dtype

from reciprocalspaceship.dtypes.base import MTZFloat32Dtype, MTZFloatArray


@register_extension_dtype
class AnomalousDifferenceDtype(MTZFloat32Dtype):
    """Dtype for anomalous difference data in reflection tables"""

    name = "AnomalousDifference"
    mtztype = "D"

    def is_friedel_dtype(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return AnomalousDifferenceArray


class AnomalousDifferenceArray(MTZFloatArray):
    """ExtensionArray for supporting AnomalousDifferenceDtype"""

    _dtype = AnomalousDifferenceDtype()
    pass
