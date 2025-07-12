# BSD 3-Clause License
#
# Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
# All rights reserved.
#
# Copyright (c) 2011-2023, Open source contributors.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import numpy as np
from pandas import Float32Dtype, Int32Dtype, Int64Dtype
from pandas._libs import lib
from pandas._typing import ArrayLike, Dtype, DtypeObj
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype
from pandas.core.dtypes.cast import np_find_common_type
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.tools.numeric import to_numeric
from pandas.util._decorators import cache_readonly

from reciprocalspaceship.dtypes.base import MTZDtype
from reciprocalspaceship.dtypes.internals import (
    BaseMaskedDtype,
    NumericArray,
    is_numeric_na,
)


class MTZInt32Dtype(MTZDtype):
    """
    An ExtensionDtype to hold a single size & kind of integer dtype.

    These specific implementations are subclasses of the non-public
    MTZInt32Dtype. For example we have Int8Dtype to represent signed int 8s.

    The attributes name & type are set when these subclasses are created.
    """

    type = np.int32

    @cache_readonly
    def is_signed_integer(self) -> bool:
        return self.kind == "i"

    @cache_readonly
    def is_unsigned_integer(self) -> bool:
        return self.kind == "u"

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        if len(set(dtypes)) == 1:
            # only itself
            return self
        # we only handle nullable EA dtypes and numeric numpy dtypes
        if not all(
            isinstance(t, BaseMaskedDtype)
            or (
                isinstance(t, np.dtype)
                and (np.issubdtype(t, np.number) or np.issubdtype(t, np.bool_))
            )
            for t in dtypes
        ):
            return None
        np_dtype = np_find_common_type(
            # error: List comprehension has incompatible type List[Union[Any,
            # dtype, ExtensionDtype]]; expected List[Union[dtype, None, type,
            # _SupportsDtype, str, Tuple[Any, Union[int, Sequence[int]]],
            # List[Any], _DtypeDict, Tuple[Any, Any]]]
            [
                (
                    t.numpy_dtype  # type: ignore[misc]
                    if isinstance(t, BaseMaskedDtype)
                    else t
                )
                for t in dtypes
            ],
            [],
        )
        if np.issubdtype(np_dtype, np.integer):
            return Int32Dtype()
        elif np.issubdtype(np_dtype, np.floating):
            return Float32Dtype()
        return None


def safe_cast(values, dtype, copy: bool):
    """
    Safely cast the values to the dtype if they
    are equivalent, meaning floats must be equivalent to the
    ints.
    """
    try:
        return values.astype(dtype, casting="safe", copy=copy)
    except TypeError as err:
        casted = values.astype(dtype, copy=copy)
        if (casted == values).all():
            return casted

        raise TypeError(
            f"cannot safely cast non-equivalent {values.dtype} to {np.dtype(dtype)}"
        ) from err


def coerce_to_array(
    values, dtype, mask=None, copy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coerce the input values array to numpy arrays with a mask.

    Parameters
    ----------
    values : 1D list-like
    dtype : integer dtype
    mask : bool 1D array, optional
    copy : bool, default False
        if True, copy the input

    Returns
    -------
    tuple of (values, mask)
    """
    # if values is integer numpy array, preserve its dtype
    if dtype is None and hasattr(values, "dtype"):
        if is_integer_dtype(values.dtype):
            dtype = values.dtype

    if dtype is not None:
        if isinstance(dtype, str) and (
            dtype.startswith("Int") or dtype.startswith("UInt")
        ):
            # Avoid DeprecationWarning from NumPy about np.dtype("Int64")
            # https://github.com/numpy/numpy/pull/7476
            dtype = dtype.lower()

        if not issubclass(type(dtype), MTZInt32Dtype):
            try:
                dtype = INT_STR_TO_DTYPE[str(np.dtype(dtype))]
            except KeyError as err:
                raise ValueError(f"invalid dtype specified {dtype}") from err

    if isinstance(values, MTZIntegerArray):
        values, mask = values._data, values._mask
        if dtype is not None:
            values = values.astype(dtype.numpy_dtype, copy=False)

        if copy:
            values = values.copy()
            mask = mask.copy()
        return values, mask

    if copy:
        values = np.array(values, copy=copy)
    else:
        values = np.asarray(values)

    inferred_type = None
    if is_object_dtype(values.dtype) or is_string_dtype(values.dtype):
        inferred_type = lib.infer_dtype(values, skipna=True)
        if inferred_type == "empty":
            pass
        elif inferred_type not in [
            "floating",
            "integer",
            "mixed-integer",
            "integer-na",
            "mixed-integer-float",
            "string",
            "unicode",
        ]:
            raise TypeError(f"{values.dtype} cannot be converted to an IntegerDtype")

    elif is_bool_dtype(values) and is_integer_dtype(dtype):
        if copy:
            values = np.array(values, dtype=int, copy=copy)
        else:
            values = np.asarray(values, dtype=int)

    elif not (is_integer_dtype(values) or is_float_dtype(values)):
        raise TypeError(f"{values.dtype} cannot be converted to an IntegerDtype")

    if values.ndim != 1:
        raise TypeError("values must be a 1D list-like")

    if mask is None:
        mask = is_numeric_na(values)
    else:
        assert len(mask) == len(values)

    if mask.ndim != 1:
        raise TypeError("mask must be a 1D list-like")

    # infer dtype if needed
    if dtype is None:
        dtype = np.dtype("int64")
    else:
        dtype = dtype.type

    # if we are float, let's make sure that we can
    # safely cast

    # we copy as need to coerce here
    if mask.any():
        values = values.copy()
        values[mask] = 1
    if inferred_type in ("string", "unicode"):
        # casts from str are always safe since they raise
        # a ValueError if the str cannot be parsed into an int
        values = values.astype(dtype, copy=copy)
    else:
        try:
            values = safe_cast(values, dtype, copy=False)
        except:
            # certain outputs cannot be coerced to int32
            dtype = np.dtype("float64")
            values = safe_cast(values, dtype, copy=False)

    return values, mask


class MTZIntegerArray(NumericArray):
    """
    Array of integer (optional missing) values.

    .. versionchanged:: 1.0.0

       Now uses :attr:`pandas.NA` as the missing value rather
       than :attr:`numpy.nan`.

    .. warning::

       MTZIntegerArray is currently experimental, and its API or internal
       implementation may change without warning.

    We represent an MTZIntegerArray with 2 numpy arrays:

    - data: contains a numpy integer array of the appropriate dtype
    - mask: a boolean array holding a mask on the data, True is missing

    To construct an MTZIntegerArray from generic array-like input, use
    :func:`pandas.array` with one of the integer dtypes (see examples).

    See :ref:`integer_na` for more.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d integer-dtype array.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values.
    copy : bool, default False
        Whether to copy the `values` and `mask`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    MTZIntegerArray

    Examples
    --------
    Create an MTZIntegerArray with :func:`pandas.array`.

    >>> int_array = pd.array([1, None, 3], dtype=pd.Int32Dtype())
    >>> int_array
    <MTZIntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: Int32

    String aliases for the dtypes are also available. They are capitalized.

    >>> pd.array([1, None, 3], dtype='Int32')
    <MTZIntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: Int32

    >>> pd.array([1, None, 3], dtype='UInt16')
    <MTZIntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: UInt16
    """

    # The value used to fill '_data' to avoid upcasting
    _internal_fill_value = 1
    # Fill values used for any/all
    _truthy_value = 1
    _falsey_value = 0

    @cache_readonly
    def dtype(self) -> MTZInt32Dtype:
        return self._dtype

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> MTZIntegerArray:
        values, mask = coerce_to_array(scalars, dtype=dtype, copy=copy)
        return cls(values, mask)

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy: bool = False
    ) -> MTZIntegerArray:
        scalars = to_numeric(strings, errors="raise")
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    def _coerce_to_array(self, value) -> tuple[np.ndarray, np.ndarray]:
        return coerce_to_array(value, dtype=self.dtype)

    def _maybe_mask_result(self, result, mask):
        """
        Parameters
        ----------
        result : array-like
        mask : array-like bool
        """
        if result.dtype.kind in "iu":
            return type(self)(result, mask, copy=False)
        return super()._maybe_mask_result(result=result, mask=mask)

    def astype(self, dtype, copy: bool = True) -> ArrayLike:
        """
        Cast to a NumPy array or ExtensionArray with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        ndarray or ExtensionArray
            NumPy ndarray, BooleanArray or MTZIntegerArray with 'dtype' for its dtype.

        Raises
        ------
        TypeError
            if incompatible type with an IntegerDtype, equivalent of same_kind
            casting
        """
        dtype = pandas_dtype(dtype)

        if isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)

        na_value: float | lib.NoDefault

        # coerce
        if is_float_dtype(dtype):
            # In astype, we consider dtype=float to also mean na_value=np.nan
            na_value = np.nan
        else:
            na_value = lib.no_default

        return self.to_numpy(dtype=dtype, na_value=na_value, copy=False)

    def _values_for_argsort(self) -> np.ndarray:
        """
        Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort : Return the indices that would sort this array.
        """
        data = self._data.copy()
        if self._mask.any():
            data[self._mask] = data.min() - 1
        return data

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


# create the Dtypes


@register_extension_dtype
class BatchDtype(MTZInt32Dtype):
    name = "Batch"
    mtztype = "B"

    def is_friedel_dtype(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return BatchArray


class BatchArray(MTZIntegerArray):
    """ExtensionArray for supporting BatchDtype"""

    _dtype = BatchDtype()
    pass


@register_extension_dtype
class HKLIndexDtype(MTZInt32Dtype):
    """Custom MTZ Dtype for Miller indices"""

    name = "HKL"
    mtztype = "H"

    def is_friedel_dtype(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return HKLIndexArray


class HKLIndexArray(MTZIntegerArray):
    _dtype = HKLIndexDtype()
    pass


@register_extension_dtype
class M_IsymDtype(MTZInt32Dtype):
    """Dtype for representing M/ISYM values"""

    name = "M/ISYM"
    mtztype = "Y"

    def is_friedel_dtype(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return M_IsymArray


class M_IsymArray(MTZIntegerArray):
    """ExtensionArray for supporting M_IsymDtype"""

    _dtype = M_IsymDtype()
    pass


@register_extension_dtype
class MTZIntDtype(MTZInt32Dtype):
    """Dtype for generic integer data"""

    name = "MTZInt"
    mtztype = "I"

    def is_friedel_dtype(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return MTZIntArray


class MTZIntArray(MTZIntegerArray):
    _dtype = MTZIntDtype()
    pass


INT_STR_TO_DTYPE: dict[str, MTZInt32Dtype] = {
    "Batch": BatchDtype(),
    "HKL": HKLIndexDtype(),
    "M/ISYM": M_IsymDtype(),
    "MTZInt": MTZIntDtype(),
    "int32": Int32Dtype(),
    "int64": Int64Dtype(),
}
