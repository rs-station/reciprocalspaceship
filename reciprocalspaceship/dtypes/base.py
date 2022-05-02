import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.floating import coerce_to_array as coerce_to_float_array
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.integer import coerce_to_array as coerce_to_int_array
from pandas.core.dtypes.common import (
    is_float,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)
from pandas.util._decorators import cache_readonly


class MTZDtype(ExtensionDtype):
    """Base ExtensionDtype for implementing persistent MTZ data types"""

    def is_friedel_dtype(self):
        """Returns whether MTZ dtype represents a Friedel dtype"""
        raise NotImplementedError

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
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
    """Base ExtensionArray class for integer arrays backed by pd.IntegerArray"""

    def _maybe_mask_result(self, result, mask, other, op_name: str):
        """
        Parameters
        ----------
        result : array-like
        mask : array-like bool
        other : scalar or array-like
        op_name : str
        """
        if is_integer_dtype(result):
            return type(self)(result, mask, copy=False)
        return super()._maybe_mask_result(
            result=result, mask=mask, other=other, op_name=op_name
        )

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

    def to_numpy(self, dtype=None, copy=False, **kwargs):
        """
        Convert to a NumPy Array.

        If `dtype` is None and array does not contain any NaNs, this method
        will return a np.int32 array.  Otherwise it will return a ndarray of
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

        Returns
        -------
        numpy.ndarray
        """
        if dtype is None and not self._hasna:
            dtype = np.int32

        # na_value is hard-coded to np.nan -- this prevents other functions
        # from resetting it.
        return super().to_numpy(dtype=dtype, copy=copy, na_value=np.nan)

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
    """Base ExtensionArray class for floating point arrays backed by pd.FloatingArray"""

    def _maybe_mask_result(self, result, mask, other, op_name: str):
        """
        Parameters
        ----------
        result : array-like
        mask : array-like bool
        other : scalar or array-like
        op_name : str
        """
        # if we have a float operand we are by-definition
        # a float result
        # or our op is a divide
        if (
            (is_float_dtype(other) or is_float(other))
            or (op_name in ["rtruediv", "truediv"])
            or (is_float_dtype(self.dtype) and is_numeric_dtype(result.dtype))
        ):
            return type(self)(result, mask, copy=False)
        return super()._maybe_mask_result(
            result=result, mask=mask, other=other, op_name=op_name
        )

    @cache_readonly
    def dtype(self):
        return self._dtype

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        values, mask = coerce_to_float_array(scalars, dtype=dtype, copy=copy)
        return cls(values, mask)

    def _coerce_to_array(self, value):
        return coerce_to_float_array(value, dtype=self.dtype)

    def to_numpy(self, dtype=None, copy=False, **kwargs):
        """
        Convert to a NumPy Array.

        If `dtype` is None it will default to a float32 ndarray.

        Parameters
        ----------
        dtype : dtype, default np.float32
            The numpy dtype to return
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            the array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary. This is typically
            only possible when no missing values are present and `dtype`
            is the equivalent numpy dtype.

        Returns
        -------
        numpy.ndarray
        """
        if dtype is None:
            dtype = np.float32

        # na_value is hard-coded to np.nan -- this prevents other functions
        # from resetting it.
        return super().to_numpy(dtype=dtype, copy=copy, na_value=np.nan)

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
