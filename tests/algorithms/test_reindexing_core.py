from __future__ import annotations

from typing import Final

import gemmi
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

import reciprocalspaceship as rs
from reciprocalspaceship.algorithms._errors import PhaseAlignmentInputError
from reciprocalspaceship.algorithms.reindexing import (
    _as_asu,
    _as_intensities,
    _common_finite_index,
    _indexed_series,
    _pearson_correlation,
    _resolution_normalize,
)

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]

SPACEGROUP: Final[gemmi.SpaceGroup] = gemmi.SpaceGroup("P 21 21 2")
UNIT_CELL: Final[gemmi.UnitCell] = gemmi.UnitCell(
    30.0,
    30.0,
    50.0,
    90.0,
    90.0,
    90.0,
)


def _dataset(
    miller_indices: IntegerArray,
    values: FloatArray,
    *,
    key: str = "F",
) -> rs.DataSet:
    dataset = rs.DataSet(
        {
            "H": miller_indices[:, 0],
            "K": miller_indices[:, 1],
            "L": miller_indices[:, 2],
            key: values,
        },
        spacegroup=SPACEGROUP,
        cell=UNIT_CELL,
        merged=True,
    )
    dataset.set_index(["H", "K", "L"], inplace=True)
    dataset[key] = dataset[key].astype("SFAmplitude")
    return dataset


def test_as_asu_returns_copy_without_synthetic_m_isym() -> None:
    miller_indices = np.asarray(((1, 2, 3), (2, 1, 4)), dtype=np.int64)
    dataset = _dataset(miller_indices, np.asarray((2.0, 3.0), dtype=np.float64))

    transformed = _as_asu(dataset, gemmi.Op("x,y,z"))

    assert transformed is not dataset
    assert "M/ISYM" not in transformed
    assert "M/ISYM" not in dataset


def test_as_asu_rejects_duplicate_indices_after_mapping() -> None:
    miller_indices = np.asarray(((1, 2, 3), (-1, -2, -3)), dtype=np.int64)
    dataset = _dataset(miller_indices, np.asarray((2.0, 3.0), dtype=np.float64))

    # Regression: distinct input rows can collapse onto one merged-ASU reflection.
    with pytest.raises(PhaseAlignmentInputError, match="unique Miller indices"):
        _as_asu(dataset, gemmi.Op("x,y,z"))


def test_indexed_series_uses_miller_indices() -> None:
    miller_indices = np.asarray(((2, 1, 4), (1, 2, 3)), dtype=np.int64)
    values = np.asarray((7.0, 5.0), dtype=np.float64)
    dataset = _dataset(miller_indices, values)

    indexed = _indexed_series(dataset, data_key="F")

    assert isinstance(indexed.index, pd.MultiIndex)
    assert indexed.index.names == ["H", "K", "L"]
    np.testing.assert_allclose(indexed.to_numpy(), values)


def test_common_finite_index_uses_one_shared_intersection() -> None:
    index = pd.MultiIndex.from_tuples(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)),
        names=("H", "K", "L"),
    )
    reference = pd.Series((1.0, 2.0, np.nan, 4.0), index=index)
    first = pd.Series((5.0, 6.0, 7.0, 8.0), index=index)
    second = pd.Series((9.0, np.inf, 11.0), index=index[[0, 1, 3]])

    common = _common_finite_index(reference, (first, second))

    assert common.tolist() == [(1, 0, 0), (1, 1, 1)]


@pytest.mark.parametrize("amplitude", [False, True])
def test_as_intensities(amplitude: bool) -> None:
    values = np.asarray((2.0, 3.0), dtype=np.float64)

    intensities = _as_intensities(values, amplitude=amplitude)

    expected = values**2 if amplitude else values
    np.testing.assert_allclose(intensities, expected)


def test_resolution_normalize_scales_each_bin_mean() -> None:
    intensities = np.arange(1.0, 205.0, dtype=np.float64)
    inverse_d_squared = np.linspace(0.01, 1.0, len(intensities))

    normalized = _resolution_normalize(intensities, inverse_d_squared)

    for indices in np.array_split(np.arange(len(intensities)), 2):
        assert np.isclose(np.mean(normalized[indices]), 1.0)


def test_resolution_normalize_rejects_nonpositive_bin_mean() -> None:
    with pytest.raises(PhaseAlignmentInputError, match="positive finite mean"):
        _resolution_normalize(
            np.asarray((-1.0, 0.0, 1.0), dtype=np.float64),
            np.asarray((1.0, 2.0, 3.0), dtype=np.float64),
        )


def test_pearson_correlation_smoke() -> None:
    first = np.asarray((1.0, 2.0, 4.0), dtype=np.float64)
    second = np.asarray((2.0, 4.0, 8.0), dtype=np.float64)

    assert np.isclose(_pearson_correlation(first, second), 1.0)


def test_pearson_correlation_rejects_constant_data() -> None:
    with pytest.raises(PhaseAlignmentInputError, match="undefined"):
        _pearson_correlation(
            np.ones(3, dtype=np.float64),
            np.arange(3, dtype=np.float64),
        )
