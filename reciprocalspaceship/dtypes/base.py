import operator
import numpy as np
from pandas.core import nanops
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionScalarOpsMixin,
    take
)
from pandas.core.tools.numeric import to_numeric

class NumpyExtensionArray(ExtensionArray, ExtensionScalarOpsMixin):
    """
    Base ExtensionArray for defining a custom Pandas.ExtensionDtype that
    uses a numpy array on the backend for storing the array data.
    """
    _itemsize = 8
    ndim = 1
    can_hold_na = True
    __array_priority__ = 1000

    def __init__(self, values, copy=True, dtype=None):
        self.data = np.array(values, dtype=self._dtype.type, copy=copy)
        if isinstance(dtype, str):
            type(self._dtype).construct_array_type(dtype)
        elif dtype:
            assert isinstance(dtype, type(self._dtype))
    
    @property
    def dtype(self):
        return self._dtype

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(scalars, dtype=dtype)

    @classmethod
    def _from_sequence_of_strings(cls, strings, dtype=None, copy=False):
        scalars = to_numeric(strings, errors="raise")
        return cls._from_sequence(scalars, dtype, copy)
    
    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    @classmethod
    def _from_ndarray(cls, data, copy=False):
        return cls(data, copy=copy)

    @property
    def shape(self):
        return self.data.shape

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
            return result
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

    def __iter__(self):
        return iter(self.data)

    def _reduce(self, name, skipna=True, **kwargs):
        data = self.data

        op = getattr(nanops, 'nan' + name)
        result = op(data, axis=0, skipna=skipna)

        return result

    def value_counts(self, dropna=True):
        """
        Returns a Series containing counts of each category.
        Every category will have an entry, even those with a count of 0.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : CrystalSeries
        """
        from pandas import Index
        import reciprocalspaceship as rs

        # compute counts on the data with no nans
        mask = np.isnan(self.data)
        data = self.data[~mask]
        value_counts = Index(data).value_counts()
        array = value_counts.values

        # TODO(extension)
        # if we have allow Index to hold an ExtensionArray
        # this is easier
        index = value_counts.index.astype(object)

        # if we want nans, count the mask
        if not dropna:

            # TODO(extension)
            # appending to an Index *always* infers
            # w/o passing the dtype
            array = np.append(array, [mask.sum()])
            index = Index(
                np.concatenate(
                    [index.values, np.array([self.dtype.na_value], dtype=object)]
                ),
                dtype=object,
            )

        return rs.CrystalSeries(array, index=index)

    def _coerce_to_ndarray(self, dtype=None):
        if dtype:
            return self.data.astype(dtype)
        else:
            return self.data.astype(self.dtype.type)

    def __array__(self, dtype=None):
        return self._coerce_to_ndarray()
