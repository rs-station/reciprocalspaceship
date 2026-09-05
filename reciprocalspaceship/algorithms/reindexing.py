"""correlation-based crystallographic reindexing"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import gemmi
import numpy as np

from reciprocalspaceship.algorithms._errors import (
    PhaseAlignmentInputError,
)
from reciprocalspaceship.dataset import DataSet

if TYPE_CHECKING:
    pass


DEFAULT_MAXIMUM_OBLIQUITY: Final[float] = 1e-6


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
