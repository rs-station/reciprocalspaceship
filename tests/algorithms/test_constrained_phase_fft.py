from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Final

import gemmi
import numpy as np
import pytest
from numpy.typing import NDArray

import reciprocalspaceship as rs
from reciprocalspaceship.algorithms.phase_alignment import (
    FFT_BYTES_PER_GRID_POINT,
    _allowed_grid_origins,
    _estimated_fft_memory,
    _integer_polar_basis,
    _origin_correlation_grid,
    _origin_cosets,
    _origin_fft_local_maxima,
    _periodic_local_maxima,
    _polar_basis,
    _primitive_integer_vector,
    _rank_internal_translations,
    _rotation_constraints,
)

FULL_ROTATION_RADIANS: Final[float] = float(2.0 * np.pi)
PHASE_DATA_DIRECTORY: Final[Path] = Path(__file__).parents[1] / "data" / "fmodel"

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]


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


def test_one_dimensional_fft_rejects_unsafe_memory_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "reciprocalspaceship.algorithms.phase_alignment.MAXIMUM_FFT_MEMORY_BYTES",
        1,
    )
    miller_indices = np.asarray(((0, 0, 1), (0, 0, 2)), dtype=np.int64)
    phases = np.zeros(2, dtype=np.float64)
    weights = np.full(2, 0.5, dtype=np.float64)

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="GiB"):
        _origin_fft_local_maxima(
            np.zeros(3, dtype=np.float64),
            np.asarray(((0,), (0,), (1,)), dtype=np.int64),
            miller_indices,
            phases,
            weights,
        )


def test_bounded_memory_fft_matches_materialized_local_maxima(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random_number_generator = np.random.default_rng(seed=20260822)
    miller_indices = random_number_generator.integers(low=-3, high=4, size=(80, 3))
    phase_differences = random_number_generator.uniform(
        low=-np.pi,
        high=np.pi,
        size=len(miller_indices),
    )
    weights = random_number_generator.uniform(low=0.1, high=1.0, size=80)
    normalized_weights = weights / np.sum(weights)
    origin_coset = np.zeros(3, dtype=np.float64)
    integer_polar_basis = np.eye(3, dtype=np.int64)
    correlation_grid = _origin_correlation_grid(
        origin_coset,
        integer_polar_basis,
        miller_indices,
        phase_differences,
        normalized_weights,
    )
    expected_indices = _periodic_local_maxima(correlation_grid)
    monkeypatch.setattr(
        "reciprocalspaceship.algorithms.phase_alignment.MAXIMUM_FFT_MEMORY_BYTES",
        5_000,
    )

    actual_indices, grid_shape = _origin_fft_local_maxima(
        origin_coset,
        integer_polar_basis,
        miller_indices,
        phase_differences,
        normalized_weights,
    )

    assert grid_shape == correlation_grid.shape
    assert {tuple(index) for index in actual_indices} == {
        tuple(index) for index in expected_indices
    }


@pytest.mark.parametrize(
    "maximum_fft_memory_bytes",
    [1_000_000, 20_000],
    ids=["materialized", "bounded-memory"],
)
def test_origin_fft_retains_strongest_local_maxima(
    monkeypatch: pytest.MonkeyPatch,
    maximum_fft_memory_bytes: int,
) -> None:
    random_number_generator = np.random.default_rng(seed=20260823)
    maximum_refinement_starts = 11
    miller_indices = random_number_generator.integers(
        low=-6,
        high=7,
        size=(160, 3),
        dtype=np.int64,
    )
    phase_differences = random_number_generator.uniform(
        low=-np.pi,
        high=np.pi,
        size=len(miller_indices),
    )
    weights = random_number_generator.uniform(
        low=0.1,
        high=1.0,
        size=len(miller_indices),
    )
    normalized_weights = weights / np.sum(weights)
    origin_coset = np.asarray((0.125, 0.25, 0.375), dtype=np.float64)
    integer_polar_basis = np.eye(3, dtype=np.int64)
    correlation_grid = _origin_correlation_grid(
        origin_coset,
        integer_polar_basis,
        miller_indices,
        phase_differences,
        normalized_weights,
    )
    all_maximum_indices = _periodic_local_maxima(correlation_grid)
    all_maximum_scores = correlation_grid[tuple(all_maximum_indices.T)]
    expected_indices = all_maximum_indices[:maximum_refinement_starts]
    expected_scores = all_maximum_scores[:maximum_refinement_starts]
    assert len(all_maximum_indices) > maximum_refinement_starts
    assert (
        np.min(np.abs(np.diff(all_maximum_scores[: maximum_refinement_starts + 1])))
        > 1e-6
    )
    monkeypatch.setattr(
        "reciprocalspaceship.algorithms.phase_alignment.MAXIMUM_FFT_MEMORY_BYTES",
        maximum_fft_memory_bytes,
    )

    actual_indices, grid_shape = _origin_fft_local_maxima(
        origin_coset,
        integer_polar_basis,
        miller_indices,
        phase_differences,
        normalized_weights,
        maximum_refinement_starts=maximum_refinement_starts,
    )

    # Regression: both FFT paths must cap work without discarding stronger peaks.
    assert grid_shape == correlation_grid.shape
    assert len(actual_indices) == maximum_refinement_starts
    np.testing.assert_array_equal(actual_indices, expected_indices)
    actual_scores = correlation_grid[tuple(actual_indices.T)]
    np.testing.assert_array_equal(actual_scores, expected_scores)


def test_rank_internal_translations_rejects_empty_origin_set() -> None:
    miller_indices = np.eye(3, dtype=np.int64)
    phases = np.zeros(3, dtype=np.float64)
    weights = np.full(3, 1.0 / 3.0, dtype=np.float64)

    with pytest.raises(
        rs.algorithms.PhaseAlignmentOptimizationError, match="no candidate"
    ):
        _rank_internal_translations(
            np.empty((0, 3), dtype=np.float64),
            np.empty((3, 0), dtype=np.int64),
            miller_indices,
            phases,
            weights,
            gemmi.SpaceGroup("P 21 21 21"),
        )


def test_origin_ranking_does_not_accept_a_stationary_saddle() -> None:
    # Regression: cos(x) - 0.3*cos(2x) has a sampled maximum but a local minimum at 0.
    # BFGS reports success there, hiding the two equally good continuous maxima.
    miller_indices = np.asarray(((1, 0, 0), (2, 0, 0)), dtype=np.int64)

    with pytest.raises(
        rs.algorithms.PhaseAlignmentOptimizationError, match="not a maximum"
    ):
        _rank_internal_translations(
            np.zeros((1, 3), dtype=np.float64),
            np.asarray(((1,), (0,), (0,)), dtype=np.int64),
            miller_indices,
            np.asarray((0.0, np.pi), dtype=np.float64),
            np.asarray((1.0, 0.3), dtype=np.float64) / 1.3,
            gemmi.SpaceGroup("P 1"),
        )


def test_per_coset_fft_recovers_noisy_p21_regression() -> None:
    dataset = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OFL.mtz"))
    miller_indices = np.asarray(dataset.get_hkls(), dtype=np.int64)
    amplitudes = dataset["FMODEL"].to_numpy(dtype=np.float64)
    normalized_weights = amplitudes**2 / np.sum(amplitudes**2)
    injected_origin_shift = np.asarray((0.5, -0.137, 0.5), dtype=np.float64)
    random_number_generator = np.random.default_rng(seed=20267843)
    phase_noise = random_number_generator.normal(
        loc=0.0,
        scale=70.0,
        size=len(miller_indices),
    )
    phase_differences = np.deg2rad(
        rs.utils.canonicalize_phases(
            360.0 * miller_indices @ injected_origin_shift + phase_noise,
        )
    )
    rotation_constraints = _rotation_constraints(dataset.spacegroup)
    polar_basis = _polar_basis(rotation_constraints)
    integer_polar_basis = _integer_polar_basis(rotation_constraints)
    allowed_grid_origins = _allowed_grid_origins(
        dataset.spacegroup,
        rotation_constraints,
    )
    origin_cosets = _origin_cosets(
        allowed_grid_origins,
        polar_basis,
        dataset.spacegroup,
    )

    candidates = _rank_internal_translations(
        origin_cosets,
        integer_polar_basis,
        miller_indices,
        phase_differences,
        normalized_weights,
        dataset.spacegroup,
    )

    # Regression: the former shared 3-D seed converged to the wrong P21 origin coset.
    assert len(origin_cosets) == 4
    best_translation, best_correlation = candidates[0]
    np.testing.assert_allclose(
        best_translation,
        (0.5, 0.136289717479, 0.5),
        atol=1e-7,
    )
    np.testing.assert_allclose(best_correlation, 0.331742292384235, atol=1e-8)
    assert best_correlation - candidates[1][1] > 0.09
