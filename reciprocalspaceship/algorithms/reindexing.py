"""correlation-based crystallographic reindexing"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Optional

import gemmi
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from reciprocalspaceship.algorithms._errors import (
    PhaseAlignmentInputError,
)
from reciprocalspaceship.dataset import DataSet
from reciprocalspaceship.utils.asu import hkl_to_asu
from reciprocalspaceship.utils.symmetry import apply_to_hkl

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


DEFAULT_MAXIMUM_OBLIQUITY: Final[float] = 1e-6
IDENTITY_OPERATION: Final[str] = "x,y,z"


TARGET_REFLECTIONS_PER_RESOLUTION_BIN: Final[int] = 100


MAXIMUM_RESOLUTION_BINS: Final[int] = 20


FloatArray: TypeAlias = NDArray[np.float64]


def _validate_maximum_obliquity(max_obliquity: float) -> float:
    try:
        validated_maximum_obliquity = float(max_obliquity)
    except (TypeError, ValueError, OverflowError) as error:
        msg = f"max_obliquity must be numeric; got {max_obliquity!r}"
        raise ValueError(msg) from error
    if (
        not np.isfinite(validated_maximum_obliquity)
        or validated_maximum_obliquity < 0.0
    ):
        msg = f"max_obliquity must be finite and nonnegative; got {max_obliquity!r}"
        raise ValueError(msg)
    return validated_maximum_obliquity


def _validate_symmetry_metadata(dataset: DataSet, *, name: str) -> None:
    if not isinstance(dataset.spacegroup, gemmi.SpaceGroup):
        msg = f"{name}.spacegroup must be set"
        raise PhaseAlignmentInputError(msg)
    if not isinstance(dataset.cell, gemmi.UnitCell):
        msg = f"{name}.cell must be set"
        raise PhaseAlignmentInputError(msg)
    cell_parameters = np.asarray(dataset.cell.parameters, dtype=np.float64)
    if (
        not np.isfinite(cell_parameters).all()
        or np.any(cell_parameters <= 0.0)
        or np.any(cell_parameters[3:] >= 180.0)
        or not np.isfinite(dataset.cell.volume)
        or dataset.cell.volume <= 0.0
    ):
        msg = f"{name}.cell must have valid geometry; got {dataset.cell.parameters!r}"
        raise PhaseAlignmentInputError(msg)


def has_reindexing_ambiguity(
    dataset: DataSet,
    *,
    max_obliquity: float = DEFAULT_MAXIMUM_OBLIQUITY,
) -> bool:
    """Test whether a dataset admits an alternative indexing operation.

    Parameters
    ----------
    dataset : DataSet
        Dataset supplying both the space group and unit-cell metric.
    max_obliquity : float, optional
        Maximum lattice-symmetry obliquity in degrees. The default includes only
        exact merohedral ambiguities; larger values include pseudo-merohedry.

    Returns
    -------
    bool
        Whether Gemmi finds at least one nonredundant proper reindexing operation.

    Raises
    ------
    ValueError
        If crystallographic metadata or ``max_obliquity`` is invalid.

    Notes
    -----
    Indexing ambiguity depends on the unit-cell metric as well as the space group.
    Hand inversion is deliberately excluded, matching Gemmi and Pointless.
    """
    if not isinstance(dataset, DataSet):
        msg = f"dataset must be an rs.DataSet; got {type(dataset).__name__}"
        raise ValueError(msg)
    _validate_symmetry_metadata(dataset, name="dataset")
    validated_maximum_obliquity = _validate_maximum_obliquity(max_obliquity)
    return bool(
        dataset.find_twin_laws(
            max_obliq=validated_maximum_obliquity,
            all_ops=False,
        )
    )


def _as_asu(dataset: DataSet, operation: gemmi.Op) -> DataSet:
    had_m_isym = "M/ISYM" in dataset
    transformed = (
        dataset
        if operation == gemmi.Op(IDENTITY_OPERATION)
        else dataset.apply_symop(operation)
    ).hkl_to_asu()
    for key in dataset.get_complex_keys():
        transformed[key] = transformed[key].astype(dataset.dtypes[key])
    if not had_m_isym and "M/ISYM" in transformed:
        transformed.drop(columns="M/ISYM", inplace=True)
    miller_indices = transformed.get_hkls()
    if len(np.unique(miller_indices, axis=0)) != len(miller_indices):
        msg = "merged data must contain unique Miller indices after mapping to the ASU"
        raise PhaseAlignmentInputError(msg)
    return transformed


def _indexed_series(
    dataset: DataSet,
    *,
    data_key: str,
    operation: Optional[gemmi.Op] = None,
) -> pd.Series[float]:
    miller_indices = dataset.get_hkls()
    if operation is not None:
        miller_indices, _ = hkl_to_asu(
            apply_to_hkl(miller_indices, operation), dataset.spacegroup
        )
    index = pd.MultiIndex.from_arrays(
        miller_indices.T,
        names=("H", "K", "L"),
    )
    if not index.is_unique:
        msg = "merged data must contain unique Miller indices after mapping to the ASU"
        raise PhaseAlignmentInputError(msg)
    return pd.Series(
        dataset[data_key].to_numpy(dtype=np.float64),
        index=index,
        dtype=np.float64,
    )


def _common_finite_index(
    reference_values: pd.Series[float],
    candidate_values: tuple[pd.Series[float], ...],
) -> pd.MultiIndex:
    common_index = reference_values.index
    for values in candidate_values:
        common_index = common_index.intersection(values.index, sort=False)
    finite = np.isfinite(reference_values.loc[common_index].to_numpy(dtype=np.float64))
    for values in candidate_values:
        finite &= np.isfinite(values.loc[common_index].to_numpy(dtype=np.float64))
    return common_index[finite]


def _as_intensities(values: FloatArray, *, amplitude: bool) -> FloatArray:
    return np.asarray(values**2 if amplitude else values, dtype=np.float64)


def _resolution_normalize(
    intensities: FloatArray,
    inverse_d_squared: FloatArray,
) -> FloatArray:
    number_of_bins = min(
        MAXIMUM_RESOLUTION_BINS,
        max(1, len(intensities) // TARGET_REFLECTIONS_PER_RESOLUTION_BIN),
    )
    resolution_order = np.argsort(inverse_d_squared)
    normalized = np.empty_like(intensities, dtype=np.float64)
    for bin_indices in np.array_split(resolution_order, number_of_bins):
        scale = float(np.mean(intensities[bin_indices]))
        if not np.isfinite(scale) or scale <= 0.0:
            msg = "intensities must have a positive finite mean in every resolution bin"
            raise PhaseAlignmentInputError(msg)
        normalized[bin_indices] = intensities[bin_indices] / scale
    return normalized


def _pearson_correlation(first: FloatArray, second: FloatArray) -> float:
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2)))
    if not np.isfinite(denominator) or denominator == 0.0:
        msg = "correlation is undefined for constant or nonfinite data"
        raise PhaseAlignmentInputError(msg)
    return float(first_centered @ second_centered / denominator)
