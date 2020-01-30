from pandas.core.dtypes.dtypes import register_extension_dtype
from .base import MTZInt32Dtype, MTZIntegerArray

@register_extension_dtype
class MTZIntDtype(MTZInt32Dtype):
    """Dtype for generic integer data"""
    name = 'MTZInt'
    mtztype = 'I'

    @classmethod
    def construct_array_type(cls):
        return MTZIntArray

class MTZIntArray(MTZIntegerArray):
    _dtype = MTZIntDtype()
    pass


