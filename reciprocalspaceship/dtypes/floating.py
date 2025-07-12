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
from pandas._libs import lib
from pandas._typing import ArrayLike, DtypeObj
from pandas.util._decorators import cache_readonly

try:
    from pandas.core.dtypes.cast import astype_nansafe
except:
    from pandas.core.dtypes.astype import _astype_nansafe as astype_nansafe

from pandas import Float32Dtype, Float64Dtype
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.cast import np_find_common_type
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_float,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import ExtensionDtype, register_extension_dtype
from pandas.core.tools.numeric import to_numeric

from reciprocalspaceship.dtypes.base import MTZDtype
from reciprocalspaceship.dtypes.internals import NumericArray, is_numeric_na


class MTZFloat32Dtype(MTZDtype):
    """
    An ExtensionDtype to hold a single size of floating dtype.

    These specific implementations are subclasses of the non-public
    MTZFloat32Dtype. For example we have Float32Dtype to represent float32.

    The attributes name & type are set when these subclasses are created.
    """

    type = np.float32

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # for now only handle other floating types
        if len(set(dtypes)) == 1:
            # only itself
            return self
        if not all(isinstance(t, MTZFloat32Dtype) for t in dtypes):
            return None
        np_dtype = np_find_common_type(
            # error: Item "ExtensionDtype" of "Union[Any, ExtensionDtype]" has no
            # attribute "numpy_dtype"
            [t.numpy_dtype for t in dtypes],  # type: ignore[union-attr]
            [],
        )
        if np.issubdtype(np_dtype, np.floating):
            return Float32Dtype()
        return None


def coerce_to_array(
    values, dtype=None, mask=None, copy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coerce the input values array to numpy arrays with a mask.

    Parameters
    ----------
    values : 1D list-like
    dtype : float dtype
    mask : bool 1D array, optional
    copy : bool, default False
        if True, copy the input

    Returns
    -------
    tuple of (values, mask)
    """
    # if values is floating numpy array, preserve its dtype
    if dtype is None and hasattr(values, "dtype"):
        if is_float_dtype(values.dtype):
            dtype = values.dtype

    if dtype is not None:
        if isinstance(dtype, str) and dtype.startswith("Float"):
            # Avoid DeprecationWarning from NumPy about np.dtype("Float64")
            # https://github.com/numpy/numpy/pull/7476
            dtype = dtype.lower()

        if not issubclass(type(dtype), MTZFloat32Dtype):
            try:
                dtype = FLOAT_STR_TO_DTYPE[str(np.dtype(dtype))]
            except KeyError as err:
                raise ValueError(f"invalid dtype specified {dtype}") from err

    if isinstance(values, MTZFloatArray):
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

    if is_object_dtype(values.dtype):
        inferred_type = lib.infer_dtype(values, skipna=True)
        if inferred_type == "empty":
            pass
        elif inferred_type not in [
            "floating",
            "integer",
            "mixed-integer",
            "integer-na",
            "mixed-integer-float",
        ]:
            raise TypeError(f"{values.dtype} cannot be converted to a MTZFloat32Dtype")

    elif is_bool_dtype(values) and is_float_dtype(dtype):
        if copy:
            values = np.array(values, dtype=float, copy=copy)
        else:
            values = np.asarray(values, dtype=float)

    elif not (is_integer_dtype(values) or is_float_dtype(values)):
        raise TypeError(f"{values.dtype} cannot be converted to a MTZFloat32Dtype")

    if values.ndim != 1:
        raise TypeError("values must be a 1D list-like")

    if mask is None:
        mask = is_numeric_na(values)

    else:
        assert len(mask) == len(values)

    if not mask.ndim == 1:
        raise TypeError("mask must be a 1D list-like")

    # infer dtype if needed
    if dtype is None:
        dtype = np.dtype("float64")
    else:
        dtype = dtype.type

    # if we are float, let's make sure that we can
    # safely cast

    # we copy as need to coerce here
    # TODO should this be a safe cast?
    if mask.any():
        values = values.copy()
        values[mask] = np.nan
    values = values.astype(dtype, copy=False)  # , casting="safe")

    return values, mask


class MTZFloatArray(NumericArray):
    """
    Array of floating (optional missing) values.

    .. versionadded:: 1.2.0

    .. warning::

       MTZFloatArray is currently experimental, and its API or internal
       implementation may change without warning. Especially the behaviour
       regarding NaN (distinct from NA missing values) is subject to change.

    We represent a MTZFloatArray with 2 numpy arrays:

    - data: contains a numpy float array of the appropriate dtype
    - mask: a boolean array holding a mask on the data, True is missing

    To construct an MTZFloatArray from generic array-like input, use
    :func:`pandas.array` with one of the float dtypes (see examples).

    See :ref:`integer_na` for more.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d float-dtype array.
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
    MTZFloatArray

    Examples
    --------
    Create an MTZFloatArray with :func:`pandas.array`:

    >>> pd.array([0.1, None, 0.3], dtype=pd.Float32Dtype())
    <MTZFloatArray>
    [0.1, <NA>, 0.3]
    Length: 3, dtype: Float32

    String aliases for the dtypes are also available. They are capitalized.

    >>> pd.array([0.1, None, 0.3], dtype="Float32")
    <MTZFloatArray>
    [0.1, <NA>, 0.3]
    Length: 3, dtype: Float32
    """

    # The value used to fill '_data' to avoid upcasting
    _internal_fill_value = 0.0
    # Fill values used for any/all
    _truthy_value = 1.0
    _falsey_value = 0.0

    @cache_readonly
    def dtype(self) -> MTZFloat32Dtype:
        return self._dtype

    def __init__(self, values: np.ndarray, mask: np.ndarray, copy: bool = False):
        if not (isinstance(values, np.ndarray) and values.dtype.kind == "f"):
            raise TypeError(
                "values should be floating numpy array. Use "
                "the 'pd.array' function instead"
            )
        if values.dtype == np.float16:
            # If we don't raise here, then accessing self.dtype would raise
            raise TypeError("MTZFloatArray does not support np.float16 dtype.")

        super().__init__(values, mask, copy=copy)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype=None, copy: bool = False
    ) -> MTZFloatArray:
        values, mask = coerce_to_array(scalars, dtype=dtype, copy=copy)
        return cls(values, mask)

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype=None, copy: bool = False
    ) -> MTZFloatArray:
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
        # if we have a float operand we are by-definition
        # a float result
        # or our op is a divide
        if result.dtype.kind == "f":
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
            NumPy ndarray, or BooleanArray, IntegerArray or MTZFloatArray with
            'dtype' for its dtype.

        Raises
        ------
        TypeError
            if incompatible type with an MTZFloat32Dtype, equivalent of same_kind
            casting
        """
        dtype = pandas_dtype(dtype)

        if isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)

        # coerce
        if is_float_dtype(dtype):
            # In astype, we consider dtype=float to also mean na_value=np.nan
            kwargs = {"na_value": np.nan}
        else:
            kwargs = {}

        # error: Argument 2 to "to_numpy" of "BaseMaskedArray" has incompatible
        # type "**Dict[str, float]"; expected "bool"
        data = self.to_numpy(dtype=dtype, **kwargs)  # type: ignore[arg-type]
        return astype_nansafe(data, dtype, copy=False)

    def _values_for_argsort(self) -> np.ndarray:
        return self._data

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


# create the Dtype
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


@register_extension_dtype
class MTZRealDtype(MTZFloat32Dtype):
    """Dtype for generic MTZ real data"""

    name = "MTZReal"
    mtztype = "R"

    def is_friedel_dtype(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return MTZRealArray


class MTZRealArray(MTZFloatArray):
    """ExtensionArray for supporting MtzRealDtype"""

    _dtype = MTZRealDtype()
    pass


@register_extension_dtype
class PhaseDtype(MTZFloat32Dtype):
    """Dtype for representing phase data in reflection tables"""

    name = "Phase"
    mtztype = "P"

    def is_friedel_dtype(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return PhaseArray


class PhaseArray(MTZFloatArray):
    """ExtensionArray for supporting PhaseDtype"""

    _dtype = PhaseDtype()
    pass


@register_extension_dtype
class HendricksonLattmanDtype(MTZFloat32Dtype):
    """
    Dtype for representing phase probability coefficients
    (Hendrickson-Lattman) in reflection tables
    """

    name = "HendricksonLattman"
    mtztype = "A"

    def is_friedel_dtype(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return HendricksonLattmanArray


class HendricksonLattmanArray(MTZFloatArray):
    """ExtensionArray for supporting HendricksonLattmanDtype"""

    _dtype = HendricksonLattmanDtype()
    pass


@register_extension_dtype
class StandardDeviationDtype(MTZFloat32Dtype):
    """Dtype for standard deviation of observables: J, F, D or other"""

    name = "Stddev"
    mtztype = "Q"

    def is_friedel_dtype(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return StandardDeviationArray


class StandardDeviationArray(MTZFloatArray):
    """ExtensionArray for supporting StandardDeviationDtype"""

    _dtype = StandardDeviationDtype()
    pass


@register_extension_dtype
class StandardDeviationFriedelSFDtype(MTZFloat32Dtype):
    """Dtype for standard deviation of F(+) or F(-)"""

    name = "StddevFriedelSF"
    mtztype = "L"

    def is_friedel_dtype(self):
        return True

    @classmethod
    def construct_array_type(cls):
        return StandardDeviationFriedelSFArray


class StandardDeviationFriedelSFArray(MTZFloatArray):
    """ExtensionArray for supporting StandardDeviationFriedelSFDtype"""

    _dtype = StandardDeviationFriedelSFDtype()
    pass


@register_extension_dtype
class StandardDeviationFriedelIDtype(MTZFloat32Dtype):
    """Dtype for standard deviation of I(+) or I(-)"""

    name = "StddevFriedelI"
    mtztype = "M"

    def is_friedel_dtype(self):
        return True

    @classmethod
    def construct_array_type(cls):
        return StandardDeviationFriedelIArray


class StandardDeviationFriedelIArray(MTZFloatArray):
    """ExtensionArray for supporting StandardDeviationFriedelIDtype"""

    _dtype = StandardDeviationFriedelIDtype()
    pass


@register_extension_dtype
class StructureFactorAmplitudeDtype(MTZFloat32Dtype):
    """Dtype for structure factor amplitude  data"""

    name = "SFAmplitude"
    mtztype = "F"

    def is_friedel_dtype(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return StructureFactorAmplitudeArray


class StructureFactorAmplitudeArray(MTZFloatArray):
    """ExtensionArray for supporting StructureFactorAmplitudeDtype"""

    _dtype = StructureFactorAmplitudeDtype()
    pass


@register_extension_dtype
class FriedelStructureFactorAmplitudeDtype(MTZFloat32Dtype):
    """
    Dtype for structure factor amplitude data from Friedel pairs --
    F(+) or F(-)
    """

    name = "FriedelSFAmplitude"
    mtztype = "G"

    def is_friedel_dtype(self):
        return True

    @classmethod
    def construct_array_type(cls):
        return FriedelStructureFactorAmplitudeArray


class FriedelStructureFactorAmplitudeArray(MTZFloatArray):
    """ExtensionArray for supporting FriedelStructureFactorAmplitudeDtype"""

    _dtype = FriedelStructureFactorAmplitudeDtype()
    pass


@register_extension_dtype
class NormalizedStructureFactorAmplitudeDtype(MTZFloat32Dtype):
    """Dtype for normalized structure factor amplitude data"""

    name = "NormalizedSFAmplitude"
    mtztype = "E"

    def is_friedel_dtype(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return NormalizedStructureFactorAmplitudeArray


class NormalizedStructureFactorAmplitudeArray(MTZFloatArray):
    """ExtensionArray for supporting NormalizedStructureFactorAmplitudeDtype"""

    _dtype = NormalizedStructureFactorAmplitudeDtype()
    pass


@register_extension_dtype
class WeightDtype(MTZFloat32Dtype):
    """Dtype for representing weights"""

    name = "Weight"
    mtztype = "W"

    def is_friedel_dtype(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return WeightArray


class WeightArray(MTZFloatArray):
    """ExtensionArray for supporting WeightDtype"""

    _dtype = WeightDtype()
    pass


FLOAT_STR_TO_DTYPE = {
    "AnomalousDifference": AnomalousDifferenceDtype(),
    "Intensity": IntensityDtype(),
    "FriedelIntensity": FriedelIntensityDtype(),
    "MTZReal": MTZRealDtype(),
    "Phase": PhaseDtype(),
    "HendricksonLattman": HendricksonLattmanDtype(),
    "Stddev": StandardDeviationDtype(),
    "StddevFriedelSF": StandardDeviationFriedelSFDtype(),
    "StddevFriedelI": StandardDeviationFriedelIDtype(),
    "SFAmplitude": StructureFactorAmplitudeDtype(),
    "FriedelSFAmplitude": FriedelStructureFactorAmplitudeDtype(),
    "NormalizedSFAmplitude": NormalizedStructureFactorAmplitudeDtype(),
    "Weight": WeightDtype(),
    "float32": Float32Dtype(),
    "float64": Float64Dtype(),
}
