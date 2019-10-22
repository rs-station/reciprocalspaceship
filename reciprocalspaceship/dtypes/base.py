import operator
import numpy as np
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionScalarOpsMixin,
    take
)

class NumpyExtensionArray(ExtensionArray, ExtensionScalarOpsMixin):
    """
    Base ExtensionArray for defining a custom Pandas.ExtensionDtype that
    uses a numpy array on the backend for storing the array data.
    """
    _itemsize = 8
    ndim = 1
    can_hold_na = True
    
    @property
    def dtype(self):
        return self._dtype

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(scalars, dtype=dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    @classmethod
    def _from_ndarray(cls, data, copy=False):
        return cls(data, copy=copy)

    @property
    def shape(self):
        return (len(self.data),)

    @property
    def na_value(self):
        return self.dtype.na_value
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, *args):
        result = operator.getitem(self.data, *args)
        if isinstance(result, tuple):
            return self._box_scalar(result)
        elif result.ndim == 0:
            return self._box_scalar(result.item())
        else:
            return type(self)(result)

    def setitem(self, indexer, value):
        """Set the 'value' inplace.
        """
        # I think having a separate than __setitem__ is good
        # since we have to return here, but __setitem__ doesn't.
        self[indexer] = value
        return self

    @property
    def nbytes(self):
        return self._itemsize * len(self)

    def _formatter(self, boxed=False):
        def fmt(x):
            if np.isnan(x):
                return "NaN"
            return str(x)
        return fmt

    def copy(self, deep=False):
        return type(self)(self.data.copy())

    def __setitem__(self, key, value):
        self.data[key] = value

    def isna(self):
        return np.isnan(self.data)

    def take(self, indexer, allow_fill=False, fill_value=None):
        if fill_value is None:
            fill_value = 0
        took = take(self.data, indexer, allow_fill=allow_fill,
                    fill_value=fill_value)
        return type(self)(took)

    @staticmethod
    def _box_scalar(scalar):
        return scalar

    def astype(self, dtype, copy=True):
        if isinstance(dtype, type(self.dtype)):
            if copy:
                self = self.copy()
            return self
        return super().astype(dtype, copy)
    
    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls(np.concatenate([array.data for array in to_concat]))

    def tolist(self):
        return self.data.tolist()

    def argsort(self, axis=-1, kind='quicksort', order=None):
        return self.data.argsort()

    def unique(self):
        _, indices = np.unique(self.data, return_index=True)
        data = self.data.take(np.sort(indices))
        return self._from_ndarray(data)


    
