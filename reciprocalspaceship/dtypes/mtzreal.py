from pandas.core.dtypes.dtypes import register_extension_dtype

from reciprocalspaceship.dtypes.base import MTZFloat32Dtype, MTZFloatArray


@register_extension_dtype
class MTZRealDtype(MTZFloat32Dtype):
    """Dtype for generic MTZ real data"""

    name = "MTZReal"
    mtztype = "R"

    @classmethod
    def construct_array_type(cls):
        return MTZRealArray


class MTZRealArray(MTZFloatArray):
    """ExtensionArray for supporting MtzRealDtype"""

    _dtype = MTZRealDtype()
    pass
