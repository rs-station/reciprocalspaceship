from __future__ import annotations

from itertools import product
from typing import Final

import gemmi
import numpy as np
import pytest

import reciprocalspaceship as rs
from reciprocalspaceship.algorithms.phase_alignment import (
    FFT_BYTES_PER_GRID_POINT,
    _estimated_fft_memory,
    _integer_polar_basis,
    _origin_correlation_grid,
    _primitive_integer_vector,
    _rotation_constraints,
)

FULL_ROTATION_RADIANS: Final[float] = float(2.0 * np.pi)


@pytest.mark.parametrize(
    ("spacegroup", "expected_dimension"),
    [
        ("P 1", 3),
        ("P 1 m 1", 2),
        ("P 61", 1),
        ("R 3:R", 1),
        ("P 21 21 21", 0),
    ],
)
def test_integer_polar_basis_spans_exact_nullspace(
    spacegroup: str,
    expected_dimension: int,
) -> None:
    constraints = _rotation_constraints(gemmi.SpaceGroup(spacegroup))

    basis = _integer_polar_basis(constraints)

    assert basis.dtype == np.int64
    assert basis.shape == (3, expected_dimension)
    np.testing.assert_array_equal(constraints @ basis, 0)


def test_integer_polar_basis_handles_oblique_rhombohedral_axis() -> None:
    constraints = _rotation_constraints(gemmi.SpaceGroup("R 3:R"))

    basis = _integer_polar_basis(constraints)

    np.testing.assert_array_equal(np.abs(basis[:, 0]), (1, 1, 1))


def test_integer_polar_basis_handles_plane_normal_to_z() -> None:
    constraints = np.asarray(((0, 0, 2),), dtype=np.int64)

    basis = _integer_polar_basis(constraints)

    np.testing.assert_array_equal(basis, ((1, 0), (0, 1), (0, 0)))


def test_integer_polar_basis_handles_zero_vector() -> None:
    constraints = np.zeros((3, 3), dtype=np.int64)

    basis = _integer_polar_basis(constraints)

    np.testing.assert_array_equal(basis, np.eye(3, dtype=np.int64))


def test_primitive_integer_vector_preserves_zero_vector() -> None:
    zero_vector = np.zeros(3, dtype=np.int64)

    np.testing.assert_array_equal(_primitive_integer_vector(zero_vector), zero_vector)


@pytest.mark.parametrize(
    "spacegroup",
    ["P 1", "P 1 m 1", "P 61", "R 3:R", "P 21 21 21"],
)
def test_origin_correlation_grid_matches_direct_evaluation(spacegroup: str) -> None:
    # Regression: a discrete origin has a scalar grid, with no floating dimensions.
    random_number_generator = np.random.default_rng(seed=20260822)
    miller_indices = random_number_generator.integers(low=-2, high=3, size=(40, 3))
    phase_differences = random_number_generator.uniform(
        low=-np.pi,
        high=np.pi,
        size=len(miller_indices),
    )
    weights = random_number_generator.uniform(low=0.1, high=1.0, size=40)
    normalized_weights = weights / np.sum(weights)
    constraints = _rotation_constraints(gemmi.SpaceGroup(spacegroup))
    basis = _integer_polar_basis(constraints)
    origin_coset = np.asarray((0.125, 0.25, 0.375), dtype=np.float64)

    correlation_grid = _origin_correlation_grid(
        origin_coset,
        basis,
        miller_indices,
        phase_differences,
        normalized_weights,
    )

    expected = np.empty_like(correlation_grid, dtype=np.float64)
    grid_shape = np.asarray(correlation_grid.shape, dtype=np.float64)
    for grid_index in product(*(range(size) for size in correlation_grid.shape)):
        polar_coordinates = np.asarray(grid_index, dtype=np.float64) / grid_shape
        translation = origin_coset + basis @ polar_coordinates
        residuals = (
            phase_differences + FULL_ROTATION_RADIANS * miller_indices @ translation
        )
        expected[grid_index] = normalized_weights @ np.cos(residuals)

    np.testing.assert_allclose(correlation_grid, expected, atol=1e-12)


def test_fft_memory_estimate_does_not_overflow() -> None:
    # Regression: multiplying grid dimensions in int64 could report zero bytes.
    assert _estimated_fft_memory((2**22,) * 3) == 2**66 * FFT_BYTES_PER_GRID_POINT


def test_origin_correlation_grid_ignores_zero_weight_frequencies() -> None:
    miller_indices = np.asarray(((1, 0, 0), (10_000, 10_000, 10_000)), dtype=np.int64)
    phase_differences = np.zeros(2, dtype=np.float64)
    normalized_weights = np.asarray((1.0, 0.0), dtype=np.float64)

    correlation_grid = _origin_correlation_grid(
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.int64),
        miller_indices,
        phase_differences,
        normalized_weights,
    )

    assert correlation_grid.shape == (3, 1, 1)


def test_origin_correlation_grid_rejects_unsafe_memory_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "reciprocalspaceship.algorithms.phase_alignment.MAXIMUM_FFT_MEMORY_BYTES",
        1,
    )
    miller_indices = np.eye(3, dtype=np.int64)
    phases = np.zeros(3, dtype=np.float64)
    weights = np.full(3, 1.0 / 3.0, dtype=np.float64)

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="GiB"):
        _origin_correlation_grid(
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.int64),
            miller_indices,
            phases,
            weights,
        )
