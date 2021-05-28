import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas.api.extensions import ExtensionDtype
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.integer import coerce_to_array as coerce_to_int_array
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.floating import coerce_to_array as coerce_to_float_array
from pandas.util._decorators import cache_readonly
import pandas as pd

class MTZDtype(ExtensionDtype):
    """Base ExtensionDtype for implementing persistent MTZ data types"""

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        elif string != cls.name and string != cls.mtztype:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")
        return cls()


class MTZInt32Dtype(MTZDtype, pd.Int32Dtype):
    """Base ExtensionDtype class for MTZDtype backed by pd.Int32Dtype"""

    def _get_common_dtype(self, dtypes):
        if len(set(dtypes)) == 1:
            # only itself
            return self
        else:
            return super(pd.Int32Dtype, self)._get_common_dtype(dtypes)
    
    def __repr__(self):
        return self.name


class MTZIntegerArray(IntegerArray):

    @cache_readonly
    def dtype(self):
        return self._dtype

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        values, mask = coerce_to_int_array(scalars, dtype=dtype, copy=copy)
        return cls(values, mask)

    @classmethod
    def _from_factorized(cls, values, original):
        values, mask = coerce_to_int_array(values, dtype=original.dtype)
        return cls(values, mask)

    def reshape(self, *args, **kwargs):
        return self._data.reshape(*args, **kwargs)

    def to_numpy(self, dtype=None, copy=False, na_value=lib.no_default):
        """
        Convert to a NumPy Array.

        If array does not contain any NaN values, will return a np.int32
        ndarray. If array contains NaN values, will return a ndarray of 
        object dtype.

        Parameters
        ----------
        dtype : dtype, default np.int32 or np.float32
            The numpy dtype to return
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            the array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary. This is typically
            only possible when no missing values are present and `dtype`
            is the equivalent numpy dtype.
        na_value : scalar, optional
             Scalar missing value indicator to use in numpy array. Defaults
             to the native missing value indicator of this array.

        Returns
        -------
        numpy.ndarray
        """
        if na_value is lib.no_default:
            na_value = libmissing.NA

        if dtype is None:
            if self._hasna:
                dtype = object
            else:
                dtype = np.int32

        if self._hasna:
            data = self._data.astype(dtype, copy=copy)
            data[self._mask] = na_value
        else:
            data = self._data.astype(dtype, copy=copy)

        return data

    def value_counts(self, dropna=True):
        """
        Returns a DataSeries containing counts of each category.
        Every category will have an entry, even those with a count of 0.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : DataSeries
        """
        from pandas import Index
        import reciprocalspaceship as rs

        # compute counts on the data with no nans
        data = self._data[~self._mask]
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
            array = np.append(array, [self._mask.sum()])
            index = Index(
                np.concatenate(
                    [index.values, np.array([self.dtype.na_value], dtype=object)]
                ),
                dtype=object,
            )

        return rs.DataSeries(array, index=index)


class MTZFloat32Dtype(MTZDtype, pd.Float32Dtype):
    """Base ExtensionDtype class for MTZDtype backed by pd.Float32Dtype"""
    
    def _get_common_dtype(self, dtypes):
        if len(set(dtypes)) == 1:
            # only itself
            return self
        else:
            return super(pd.Float32Dtype, self)._get_common_dtype(dtypes)
    
    def __repr__(self):
        return self.name


class MTZFloatArray(FloatingArray):

    @cache_readonly
    def dtype(self):
        return self._dtype

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        values, mask = coerce_to_float_array(scalars, dtype=dtype, copy=copy)
        return cls(values, mask)

    def _coerce_to_array(self, value):
        return coerce_to_float_array(value, dtype=self.dtype)

    def to_numpy(self, dtype=None, copy=False, na_value=lib.no_default):
        """
        Convert to a NumPy Array.

        If array does not contain any NaN values, will return a np.int32
        ndarray. If array contains NaN values, will return a ndarray of 
        object dtype.

        Parameters
        ----------
        dtype : dtype, default np.int32 or np.float32
            The numpy dtype to return
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            the array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary. This is typically
            only possible when no missing values are present and `dtype`
            is the equivalent numpy dtype.
        na_value : scalar, optional
             Scalar missing value indicator to use in numpy array. Defaults
             to the native missing value indicator of this array.

        Returns
        -------
        numpy.ndarray
        """
        if na_value is lib.no_default:
            na_value = np.nan

        if dtype is None:
            dtype = np.float32

        if self._hasna:
            data = self._data.astype(dtype, copy=copy)
            data[self._mask] = na_value
        else:
            data = self._data.astype(dtype, copy=copy)

        return data
    
    def value_counts(self, dropna=True):
        """
        Returns a DataSeries containing counts of each category.
        Every category will have an entry, even those with a count of 0.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : DataSeries
        """
        from pandas import Index
        import reciprocalspaceship as rs

        # compute counts on the data with no nans
        data = self._data[~self._mask]
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
            array = np.append(array, [self._mask.sum()])
            index = Index(
                np.concatenate(
                    [index.values, np.array([self.dtype.na_value], dtype=object)]
                ),
                dtype=object,
            )

        return rs.DataSeries(array, index=index)
