"""constrained crystallographic phase alignment"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heapreplace
from itertools import product
from math import gcd, prod
from typing import TYPE_CHECKING, Final, Literal, Optional, Union

import gemmi
import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len
from scipy.ndimage import maximum_filter
from scipy.optimize import OptimizeResult, minimize

from reciprocalspaceship.algorithms._errors import (
    PhaseAlignmentInputError,
    PhaseAlignmentOptimizationError,
)
from reciprocalspaceship.algorithms.reindexing import (
    DEFAULT_MAXIMUM_OBLIQUITY,
    IDENTITY_OPERATION,
    ReindexingResult,
    _as_asu,
    _common_finite_index,
    _indexed_series,
    _validate_dataset,
    has_reindexing_ambiguity,
    reindex_by_correlation,
)
from reciprocalspaceship.dataseries import DataSeries
from reciprocalspaceship.dataset import DataSet
from reciprocalspaceship.dtypes import IntensityDtype, PhaseDtype, WeightDtype
from reciprocalspaceship.utils.phases import canonicalize_phases

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


FloatArray: TypeAlias = NDArray[np.float64]


IntegerArray: TypeAlias = NDArray[np.int64]


ComplexArray: TypeAlias = NDArray[np.complex128]


SpaceGroupLike: TypeAlias = Union[str, int, gemmi.SpaceGroup]


WeightingMode: TypeAlias = Literal["amplitude", "uniform"]


NUMBER_OF_CRYSTALLOGRAPHIC_AXES: Final[int] = 3


FULL_ROTATION_DEGREES: Final[float] = 360.0


FULL_ROTATION_RADIANS: Final[float] = float(2.0 * np.pi)


ORIGIN_DENOMINATOR: Final[int] = int(gemmi.Op.DEN)


SINGULAR_VALUE_TOLERANCE: Final[float] = 1e-10


OPTIMIZER_GRADIENT_TOLERANCE: Final[float] = 1e-7


OPTIMIZER_ACCEPTABLE_GRADIENT: Final[float] = 1e-5
OPTIMIZER_CURVATURE_TOLERANCE: Final[float] = 1e-7
OPTIMIZER_MAXIMUM_ITERATIONS: Final[int] = 500


TRANSLATION_ROUNDING_DECIMALS: Final[int] = 12


MAXIMUM_FFT_MEMORY_BYTES: Final[int] = 1_073_741_824


FFT_BYTES_PER_GRID_POINT: Final[int] = 80


FFT_SLAB_BYTES_PER_GRID_POINT: Final[int] = 80


DEFAULT_MAXIMUM_REFINEMENT_STARTS: Final[int] = 4_096


MINIMUM_PHASE_REFLECTIONS: Final[int] = 3


LOCAL_MAXIMUM_NEIGHBORHOOD: Final[int] = 3


TRANSLATION_EQUIVALENCE_TOLERANCE: Final[float] = 1e-6


POLAR_AXIS_CONSTRAINT_RANK: Final[int] = NUMBER_OF_CRYSTALLOGRAPHIC_AXES - 1


PERIODIC_OFFSETS: Final[FloatArray] = np.asarray(
    tuple(product((-1.0, 0.0, 1.0), repeat=NUMBER_OF_CRYSTALLOGRAPHIC_AXES)),
    dtype=np.float64,
)


@dataclass(frozen=True)
class OriginShiftCandidate:
    """A symmetry-allowed origin shift and its phase correlation.

    Attributes
    ----------
    origin_shift : tuple of float
        Fractional change of origin in the Phenix sign convention.
    correlation : float
        Normalized weighted mean cosine of the phase residuals.
    inverted_hand : bool
        Whether the moving phases were complex-conjugated before scoring.

    """

    origin_shift: tuple[float, float, float]
    correlation: float
    inverted_hand: bool


@dataclass(frozen=True)
class PhaseAlignmentResult:
    """An aligned dataset and the diagnostics supporting its solution.

    Attributes
    ----------
    dataset : DataSet
        Reindexed and origin-shifted copy aligned to the reference.
    origin_shift : tuple of float
        Selected fractional change of origin in the Phenix sign convention.
    inverted_hand : bool
        Whether the moving phase and complex columns were conjugated.
    correlation : float
        Correlation of the selected origin shift.
    runner_up_correlation : float, optional
        Correlation of the next crystallographically distinct candidate.
    correlation_gap : float, optional
        Best-minus-runner-up correlation, if a runner-up exists.
    candidates : tuple of OriginShiftCandidate
        All distinct candidates, sorted from highest to lowest correlation.
    reindexing : ReindexingResult, optional
        Diagnostics for the indexing operation applied before origin alignment, or
        ``None`` when the unit-cell metric has no indexing ambiguity.

    """

    dataset: DataSet
    origin_shift: tuple[float, float, float]
    inverted_hand: bool
    correlation: float
    runner_up_correlation: Optional[float]
    correlation_gap: Optional[float]
    candidates: tuple[OriginShiftCandidate, ...]
    reindexing: Optional[ReindexingResult]


def _rotation_constraints(spacegroup: gemmi.SpaceGroup) -> IntegerArray:
    identity_rotation = np.eye(NUMBER_OF_CRYSTALLOGRAPHIC_AXES, dtype=np.int64)
    constraints = [
        identity_rotation
        - np.asarray(operation.rot, dtype=np.int64) // ORIGIN_DENOMINATOR
        for operation in spacegroup.operations().sym_ops
    ]
    return np.concatenate(constraints, axis=0)


def _polar_basis(rotation_constraints: IntegerArray) -> FloatArray:
    _, singular_values, right_singular_vectors = np.linalg.svd(
        rotation_constraints.astype(np.float64),
        full_matrices=True,
    )
    rank = int(np.count_nonzero(singular_values > SINGULAR_VALUE_TOLERANCE))
    return np.asarray(right_singular_vectors[rank:].T, dtype=np.float64)


def _extended_gcd(first: int, second: int) -> tuple[int, int, int]:
    old_remainder, remainder = abs(first), abs(second)
    old_first_coefficient, first_coefficient = 1, 0
    old_second_coefficient, second_coefficient = 0, 1
    while remainder != 0:
        quotient = old_remainder // remainder
        old_remainder, remainder = (
            remainder,
            old_remainder - quotient * remainder,
        )
        old_first_coefficient, first_coefficient = (
            first_coefficient,
            old_first_coefficient - quotient * first_coefficient,
        )
        old_second_coefficient, second_coefficient = (
            second_coefficient,
            old_second_coefficient - quotient * second_coefficient,
        )
    first_sign = -1 if first < 0 else 1
    second_sign = -1 if second < 0 else 1
    return (
        old_remainder,
        old_first_coefficient * first_sign,
        old_second_coefficient * second_sign,
    )


def _primitive_integer_vector(vector: IntegerArray) -> IntegerArray:
    common_divisor = 0
    for value in vector:
        common_divisor = gcd(common_divisor, abs(int(value)))
    if common_divisor == 0:
        return vector
    return np.asarray(vector // common_divisor, dtype=np.int64)


def _integer_polar_basis(rotation_constraints: IntegerArray) -> IntegerArray:
    rank = int(np.linalg.matrix_rank(rotation_constraints.astype(np.float64)))
    if rank == 0:
        return np.eye(NUMBER_OF_CRYSTALLOGRAPHIC_AXES, dtype=np.int64)
    if rank == NUMBER_OF_CRYSTALLOGRAPHIC_AXES:
        return np.empty((NUMBER_OF_CRYSTALLOGRAPHIC_AXES, 0), dtype=np.int64)

    nonzero_rows = rotation_constraints[np.any(rotation_constraints != 0, axis=1)]
    if rank == POLAR_AXIS_CONSTRAINT_RANK:
        first_row = nonzero_rows[0]
        null_vector = next(
            cross_product
            for second_row in nonzero_rows[1:]
            if np.any((cross_product := np.cross(first_row, second_row)) != 0)
        )
        primitive_null_vector = _primitive_integer_vector(null_vector)
        return primitive_null_vector[:, None]

    primitive_normal = _primitive_integer_vector(nonzero_rows[0])
    first, second, third = (int(value) for value in primitive_normal)
    if first == 0 and second == 0:
        return np.asarray(((1, 0), (0, 1), (0, 0)), dtype=np.int64)
    common_divisor, first_coefficient, second_coefficient = _extended_gcd(
        first,
        second,
    )
    first_basis_vector = np.asarray(
        (second // common_divisor, -first // common_divisor, 0),
        dtype=np.int64,
    )
    second_basis_vector = np.asarray(
        (
            -first_coefficient * third,
            -second_coefficient * third,
            common_divisor,
        ),
        dtype=np.int64,
    )
    return np.column_stack((first_basis_vector, second_basis_vector))


def _origin_fourier_terms(
    origin_coset: FloatArray,
    integer_polar_basis: IntegerArray,
    miller_indices: IntegerArray,
    phase_differences: FloatArray,
    normalized_weights: FloatArray,
) -> tuple[tuple[int, ...], IntegerArray, ComplexArray]:
    positive_weights: NDArray[np.bool_] = normalized_weights > 0.0
    weighted_miller_indices = miller_indices[positive_weights]
    weighted_phase_differences = phase_differences[positive_weights]
    weighted_normalized_weights = normalized_weights[positive_weights]
    polar_frequencies = weighted_miller_indices @ integer_polar_basis
    maximum_frequencies = np.max(np.abs(polar_frequencies), axis=0)
    grid_shape = tuple(
        next_fast_len(int(2 * maximum_frequency + 1))
        for maximum_frequency in maximum_frequencies
    )
    wrapped_frequencies = np.asarray(
        polar_frequencies % np.asarray(grid_shape, dtype=np.int64),
        dtype=np.int64,
    )
    phase_at_coset = (
        weighted_phase_differences
        + FULL_ROTATION_RADIANS * weighted_miller_indices @ origin_coset
    )
    coefficients = np.asarray(
        weighted_normalized_weights * np.exp(1j * phase_at_coset),
        dtype=np.complex128,
    )
    return grid_shape, wrapped_frequencies, coefficients


def _estimated_fft_memory(grid_shape: tuple[int, ...]) -> int:
    return prod(grid_shape) * FFT_BYTES_PER_GRID_POINT


def _raise_fft_memory_error(estimated_memory: int, *, description: str) -> None:
    if estimated_memory > MAXIMUM_FFT_MEMORY_BYTES:
        msg = (
            f"{description} requires an estimated "
            f"{estimated_memory / 2**30:.2f} GiB, exceeding the "
            f"{MAXIMUM_FFT_MEMORY_BYTES / 2**30:.2f} GiB safety limit"
        )
        raise PhaseAlignmentInputError(msg)


def _materialized_correlation_grid(
    grid_shape: tuple[int, ...],
    wrapped_frequencies: IntegerArray,
    coefficients: ComplexArray,
) -> FloatArray:
    if not grid_shape:
        return np.asarray(coefficients.real.sum(), dtype=np.float64)
    fourier_coefficients = np.zeros(grid_shape, dtype=np.complex128)
    frequency_indices = tuple(
        wrapped_frequencies[:, axis] for axis in range(len(grid_shape))
    )
    np.add.at(
        fourier_coefficients,
        frequency_indices,
        coefficients,
    )
    number_of_grid_points = prod(grid_shape)
    correlation = np.fft.ifftn(fourier_coefficients).real * number_of_grid_points
    return np.asarray(correlation, dtype=np.float64)


def _origin_correlation_grid(
    origin_coset: FloatArray,
    integer_polar_basis: IntegerArray,
    miller_indices: IntegerArray,
    phase_differences: FloatArray,
    normalized_weights: FloatArray,
) -> FloatArray:
    grid_shape, wrapped_frequencies, coefficients = _origin_fourier_terms(
        origin_coset,
        integer_polar_basis,
        miller_indices,
        phase_differences,
        normalized_weights,
    )
    estimated_memory = _estimated_fft_memory(grid_shape)
    _raise_fft_memory_error(
        estimated_memory,
        description="constrained origin FFT",
    )
    return _materialized_correlation_grid(
        grid_shape,
        wrapped_frequencies,
        coefficients,
    )


def _allowed_grid_origins(
    spacegroup: gemmi.SpaceGroup,
    rotation_constraints: IntegerArray,
) -> FloatArray:
    grid_coordinates = (
        np.indices(
            (ORIGIN_DENOMINATOR,) * NUMBER_OF_CRYSTALLOGRAPHIC_AXES,
            dtype=np.int64,
        )
        .reshape(NUMBER_OF_CRYSTALLOGRAPHIC_AXES, -1)
        .T
    )
    centering_translations = (
        np.asarray(spacegroup.operations().cen_ops, dtype=np.int64) % ORIGIN_DENOMINATOR
    )
    allowed = np.ones(len(grid_coordinates), dtype=np.bool_)
    for constraint in rotation_constraints.reshape(
        -1,
        NUMBER_OF_CRYSTALLOGRAPHIC_AXES,
        NUMBER_OF_CRYSTALLOGRAPHIC_AXES,
    ):
        origin_shifts = grid_coordinates @ constraint.T % ORIGIN_DENOMINATOR
        allowed &= np.all(
            origin_shifts[:, None, :] == centering_translations[None, :, :],
            axis=2,
        ).any(axis=1)
    return np.asarray(
        grid_coordinates[allowed] / ORIGIN_DENOMINATOR,
        dtype=np.float64,
    )


def _origin_cosets(
    allowed_grid_origins: FloatArray,
    polar_basis: FloatArray,
    spacegroup: gemmi.SpaceGroup,
) -> FloatArray:
    polar_projection = polar_basis @ polar_basis.T
    nonpolar_projection = np.eye(NUMBER_OF_CRYSTALLOGRAPHIC_AXES) - polar_projection
    centering_translations = (
        np.asarray(spacegroup.operations().cen_ops, dtype=np.float64)
        / ORIGIN_DENOMINATOR
    )
    representatives: list[FloatArray] = []
    for origin in allowed_grid_origins:
        equivalent_to_existing = False
        for representative in representatives:
            periodic_differences = (
                origin
                - representative
                - centering_translations[:, None, :]
                + PERIODIC_OFFSETS[None, :, :]
            )
            nonpolar_components = periodic_differences @ nonpolar_projection
            if np.any(
                np.linalg.norm(nonpolar_components, axis=2) < SINGULAR_VALUE_TOLERANCE
            ):
                equivalent_to_existing = True
                break
        if not equivalent_to_existing:
            representatives.append(origin)
    return np.asarray(representatives, dtype=np.float64)


def has_origin_shift_ambiguity(spacegroup: SpaceGroupLike) -> bool:
    """Test whether a space group permits a distinguishable origin shift.

    Parameters
    ----------
    spacegroup : str, int, gemmi.SpaceGroup
        Space group to test.

    Returns
    -------
    bool
        Whether the space group has continuous origin freedom or more than one
        discrete origin class after quotienting lattice and centering translations.

    Raises
    ------
    PhaseAlignmentInputError
        If ``spacegroup`` cannot be converted to a Gemmi space group.
    """
    if isinstance(spacegroup, gemmi.SpaceGroup):
        validated_spacegroup = spacegroup
    else:
        msg = (
            f"spacegroup could not be converted to gemmi.SpaceGroup; got {spacegroup!r}"
        )
        if (
            not isinstance(spacegroup, (str, int))
            or isinstance(spacegroup, bool)
            or (isinstance(spacegroup, int) and not 1 <= spacegroup <= 230)
        ):
            raise PhaseAlignmentInputError(msg)
        try:
            validated_spacegroup = gemmi.SpaceGroup(spacegroup)
        except (TypeError, ValueError) as error:
            raise PhaseAlignmentInputError(msg) from error
    rotation_constraints = _rotation_constraints(validated_spacegroup)
    polar_basis = _polar_basis(rotation_constraints)
    if polar_basis.shape[1] > 0:
        return True
    allowed_grid_origins = _allowed_grid_origins(
        validated_spacegroup,
        rotation_constraints,
    )
    return (
        len(_origin_cosets(allowed_grid_origins, polar_basis, validated_spacegroup)) > 1
    )


def _phase_loss(
    fractional_translation: FloatArray,
    miller_indices: IntegerArray,
    phase_differences: FloatArray,
    normalized_weights: FloatArray,
) -> float:
    residuals = (
        phase_differences
        + FULL_ROTATION_RADIANS * miller_indices @ fractional_translation
    )
    return float(normalized_weights @ (1.0 - np.cos(residuals)))


def _refine_translation(
    starting_translation: FloatArray,
    polar_basis: FloatArray,
    miller_indices: IntegerArray,
    phase_differences: FloatArray,
    normalized_weights: FloatArray,
) -> tuple[FloatArray, float]:
    polar_frequencies = FULL_ROTATION_RADIANS * miller_indices @ polar_basis
    starting_residuals = (
        phase_differences
        + FULL_ROTATION_RADIANS * miller_indices @ starting_translation
    )

    def objective(polar_coordinates: FloatArray) -> tuple[float, FloatArray]:
        residuals = starting_residuals + polar_frequencies @ polar_coordinates
        weighted_sines = normalized_weights * np.sin(residuals)
        loss = float(normalized_weights @ (1.0 - np.cos(residuals)))
        polar_gradient = polar_frequencies.T @ weighted_sines
        return loss, np.asarray(polar_gradient, dtype=np.float64)

    result: OptimizeResult = minimize(
        objective,
        np.zeros(polar_basis.shape[1], dtype=np.float64),
        method="BFGS",
        jac=True,
        options={
            "gtol": OPTIMIZER_GRADIENT_TOLERANCE,
            "maxiter": OPTIMIZER_MAXIMUM_ITERATIONS,
        },
    )
    result_gradient = np.asarray(result.jac, dtype=np.float64)
    finite_result = (
        np.isfinite(result.fun)
        and np.isfinite(result_gradient).all()
        and np.isfinite(result.x).all()
    )
    if not finite_result or (
        not result.success
        and np.linalg.norm(result_gradient, ord=np.inf) > OPTIMIZER_ACCEPTABLE_GRADIENT
    ):
        msg = f"phase alignment did not converge: {result.message}"
        raise PhaseAlignmentOptimizationError(msg)

    # A sampled FFT maximum can be a continuous saddle with a zero gradient.
    residuals = starting_residuals + polar_frequencies @ result.x
    hessian = polar_frequencies.T @ (
        (normalized_weights * np.cos(residuals))[:, None] * polar_frequencies
    )
    if np.linalg.eigvalsh(hessian)[0] < -OPTIMIZER_CURVATURE_TOLERANCE:
        msg = "phase alignment converged to a stationary point that is not a maximum"
        raise PhaseAlignmentOptimizationError(msg)
    translation = (
        starting_translation + polar_basis @ np.asarray(result.x, dtype=np.float64)
    ) % 1.0
    return np.asarray(translation, dtype=np.float64), float(result.fun)


def _periodic_local_maxima(correlation_grid: FloatArray) -> IntegerArray:
    local_maximum = maximum_filter(
        correlation_grid,
        size=LOCAL_MAXIMUM_NEIGHBORHOOD,
        mode="wrap",
    )
    maximum_indices = np.argwhere(
        np.isclose(correlation_grid, local_maximum, rtol=0.0, atol=1e-14)
    )
    scores = correlation_grid[tuple(maximum_indices.T)]
    descending_score_order = np.lexsort((*maximum_indices.T[::-1], scores))[::-1]
    return np.asarray(maximum_indices[descending_score_order], dtype=np.int64)


def _bounded_memory_periodic_local_maxima(
    grid_shape: tuple[int, ...],
    wrapped_frequencies: IntegerArray,
    coefficients: ComplexArray,
    *,
    maximum_refinement_starts: int,
) -> IntegerArray:
    scan_axis = int(np.argmax(grid_shape))
    axis_order = (scan_axis,) + tuple(
        axis for axis in range(len(grid_shape)) if axis != scan_axis
    )
    permuted_grid_shape = tuple(grid_shape[axis] for axis in axis_order)
    permuted_frequencies = wrapped_frequencies[:, axis_order]
    slab_shape = permuted_grid_shape[1:]
    slab_frequencies = permuted_frequencies[:, 1:]
    inverse_axis_order = tuple(int(axis) for axis in np.argsort(axis_order))
    estimated_slab_memory = prod(slab_shape) * FFT_SLAB_BYTES_PER_GRID_POINT
    _raise_fft_memory_error(
        estimated_slab_memory,
        description="bounded-memory origin FFT slab",
    )

    def correlation_slab(scan_index: int) -> FloatArray:
        scan_axis_phases = np.exp(
            1j
            * FULL_ROTATION_RADIANS
            * permuted_frequencies[:, 0]
            * scan_index
            / permuted_grid_shape[0]
        )
        return _materialized_correlation_grid(
            slab_shape,
            slab_frequencies,
            coefficients * scan_axis_phases,
        )

    number_of_slabs = permuted_grid_shape[0]
    previous_slab = correlation_slab(number_of_slabs - 1)
    current_slab = correlation_slab(0)
    next_slab = correlation_slab(1 % number_of_slabs)
    strongest_candidates: list[tuple[float, tuple[int, ...]]] = []
    for scan_index in range(number_of_slabs):
        neighborhood_maximum = np.maximum.reduce(
            tuple(
                maximum_filter(
                    slab,
                    size=LOCAL_MAXIMUM_NEIGHBORHOOD,
                    mode="wrap",
                )
                for slab in (previous_slab, current_slab, next_slab)
            )
        )
        remaining_indices = np.argwhere(
            np.isclose(
                current_slab,
                neighborhood_maximum,
                rtol=0.0,
                atol=1e-14,
            )
        )
        for remaining_index in remaining_indices:
            permuted_index = (
                scan_index,
                *(int(value) for value in remaining_index),
            )
            index = tuple(permuted_index[axis] for axis in inverse_axis_order)
            candidate = (float(current_slab[tuple(remaining_index)]), index)
            if len(strongest_candidates) < maximum_refinement_starts:
                heappush(strongest_candidates, candidate)
            elif candidate > strongest_candidates[0]:
                heapreplace(strongest_candidates, candidate)
        if scan_index < number_of_slabs - 1:
            previous_slab, current_slab = current_slab, next_slab
            next_slab = correlation_slab((scan_index + 2) % number_of_slabs)
    ranked_candidates = sorted(strongest_candidates, reverse=True)
    return np.asarray(
        [candidate[1] for candidate in ranked_candidates],
        dtype=np.int64,
    )


def _origin_fft_local_maxima(
    origin_coset: FloatArray,
    integer_polar_basis: IntegerArray,
    miller_indices: IntegerArray,
    phase_differences: FloatArray,
    normalized_weights: FloatArray,
    *,
    maximum_refinement_starts: int = DEFAULT_MAXIMUM_REFINEMENT_STARTS,
) -> tuple[IntegerArray, tuple[int, ...]]:
    grid_shape, wrapped_frequencies, coefficients = _origin_fourier_terms(
        origin_coset,
        integer_polar_basis,
        miller_indices,
        phase_differences,
        normalized_weights,
    )
    estimated_memory = _estimated_fft_memory(grid_shape)
    if estimated_memory <= MAXIMUM_FFT_MEMORY_BYTES:
        correlation_grid = _materialized_correlation_grid(
            grid_shape,
            wrapped_frequencies,
            coefficients,
        )
        maximum_indices = _periodic_local_maxima(correlation_grid)
        return maximum_indices[:maximum_refinement_starts], grid_shape
    if len(grid_shape) < 2:
        _raise_fft_memory_error(
            estimated_memory,
            description="constrained origin FFT",
        )
    return (
        _bounded_memory_periodic_local_maxima(
            grid_shape,
            wrapped_frequencies,
            coefficients,
            maximum_refinement_starts=maximum_refinement_starts,
        ),
        grid_shape,
    )


def _equivalent_by_centering(
    first: FloatArray,
    second: FloatArray,
    spacegroup: gemmi.SpaceGroup,
) -> bool:
    centering_translations = (
        np.asarray(spacegroup.operations().cen_ops, dtype=np.float64)
        / ORIGIN_DENOMINATOR
    )
    periodic_differences = (
        first
        - second
        - centering_translations[:, None, :]
        + PERIODIC_OFFSETS[None, :, :]
    )
    return bool(
        np.any(
            np.linalg.norm(periodic_differences, axis=2)
            < TRANSLATION_EQUIVALENCE_TOLERANCE
        )
    )


def _rank_internal_translations(
    origin_cosets: FloatArray,
    integer_polar_basis: IntegerArray,
    miller_indices: IntegerArray,
    phase_differences: FloatArray,
    normalized_weights: FloatArray,
    spacegroup: gemmi.SpaceGroup,
    *,
    maximum_refinement_starts: int = DEFAULT_MAXIMUM_REFINEMENT_STARTS,
) -> tuple[tuple[FloatArray, float], ...]:
    polar_dimension = integer_polar_basis.shape[1]
    floating_polar_basis = integer_polar_basis.astype(np.float64)
    if polar_dimension > 0:
        identifiable_rank = np.linalg.matrix_rank(
            miller_indices[normalized_weights > 0.0] @ integer_polar_basis,
            tol=SINGULAR_VALUE_TOLERANCE,
        )
        if identifiable_rank != polar_dimension:
            msg = (
                "positive-weight reflections do not identify every continuous "
                "origin direction"
            )
            raise PhaseAlignmentInputError(msg)

    refined_candidates: list[tuple[FloatArray, float]] = []
    optimization_errors: list[PhaseAlignmentOptimizationError] = []
    for origin_coset in origin_cosets:
        if polar_dimension == 0:
            refined_candidates.append(
                (
                    origin_coset,
                    1.0
                    - _phase_loss(
                        origin_coset,
                        miller_indices,
                        phase_differences,
                        normalized_weights,
                    ),
                )
            )
            continue
        maximum_indices, integer_grid_shape = _origin_fft_local_maxima(
            origin_coset,
            integer_polar_basis,
            miller_indices,
            phase_differences,
            normalized_weights,
            maximum_refinement_starts=maximum_refinement_starts,
        )
        grid_shape = np.asarray(integer_grid_shape, dtype=np.float64)
        for maximum_index in maximum_indices:
            starting_translation = (
                origin_coset + integer_polar_basis @ (maximum_index / grid_shape)
            ) % 1.0
            try:
                translation, loss = _refine_translation(
                    starting_translation,
                    floating_polar_basis,
                    miller_indices,
                    phase_differences,
                    normalized_weights,
                )
            except PhaseAlignmentOptimizationError as error:
                optimization_errors.append(error)
                continue
            refined_candidates.append((translation, 1.0 - loss))
    if not refined_candidates:
        if optimization_errors:
            raise optimization_errors[0]
        msg = "origin search produced no candidate translations"
        raise PhaseAlignmentOptimizationError(msg)

    ranked_candidates = sorted(
        refined_candidates,
        key=lambda candidate: candidate[1],
        reverse=True,
    )
    inequivalent_candidates: list[tuple[FloatArray, float]] = []
    for translation, correlation in ranked_candidates:
        if any(
            _equivalent_by_centering(translation, existing[0], spacegroup)
            for existing in inequivalent_candidates
        ):
            continue
        inequivalent_candidates.append((translation, float(correlation)))
    return tuple(inequivalent_candidates)


def _phenix_origin_shift(
    internal_translation: FloatArray,
) -> tuple[float, float, float]:
    origin_shift = (-internal_translation + 0.5) % 1.0 - 0.5
    rounded = np.round(origin_shift, decimals=TRANSLATION_ROUNDING_DECIMALS)
    return float(rounded[0]), float(rounded[1]), float(rounded[2])


def _validate_phase_key(dataset: DataSet, *, phase_key: str, name: str) -> None:
    if phase_key not in dataset:
        msg = f"{name} does not contain phase key {phase_key!r}"
        raise PhaseAlignmentInputError(msg)
    if not isinstance(dataset.dtypes[phase_key], PhaseDtype):
        msg = f"{name}[{phase_key!r}] must have a Phase MTZ dtype"
        raise PhaseAlignmentInputError(msg)


def _validate_fom_keys(
    dataset: DataSet,
    reference: DataSet,
    *,
    fom_key: Optional[str],
    reference_fom_key: Optional[str],
) -> None:
    if (fom_key is None) != (reference_fom_key is None):
        msg = "fom_key and reference_fom_key must either both be set or both be None"
        raise PhaseAlignmentInputError(msg)
    for name, current_dataset, key in (
        ("dataset", dataset, fom_key),
        ("reference", reference, reference_fom_key),
    ):
        if key is None:
            continue
        if key not in current_dataset:
            msg = f"{name} does not contain FOM key {key!r}"
            raise PhaseAlignmentInputError(msg)
        if not isinstance(current_dataset.dtypes[key], WeightDtype):
            msg = f"{name}[{key!r}] must have a Weight MTZ dtype"
            raise PhaseAlignmentInputError(msg)


def _matched_phase_data(
    dataset: DataSet,
    reference: DataSet,
    *,
    phase_key: str,
    reference_phase_key: str,
    amplitude_key: str,
    reference_amplitude_key: str,
    fom_key: Optional[str],
    reference_fom_key: Optional[str],
    weighting: WeightingMode,
) -> tuple[IntegerArray, FloatArray, FloatArray, FloatArray]:
    reference_asu = _as_asu(reference, gemmi.Op(IDENTITY_OPERATION))
    moving_phase = _indexed_series(dataset, data_key=phase_key)
    reference_phase = _indexed_series(reference_asu, data_key=reference_phase_key)
    moving_amplitude = _indexed_series(dataset, data_key=amplitude_key)
    reference_amplitude = _indexed_series(
        reference_asu,
        data_key=reference_amplitude_key,
    )
    indexed_values = [
        moving_phase,
        reference_phase,
        moving_amplitude,
        reference_amplitude,
    ]
    moving_fom = None
    reference_fom = None
    if fom_key is not None and reference_fom_key is not None:
        moving_fom = _indexed_series(dataset, data_key=fom_key)
        reference_fom = _indexed_series(reference_asu, data_key=reference_fom_key)
        indexed_values.extend((moving_fom, reference_fom))

    common_index = _common_finite_index(
        indexed_values[0],
        tuple(indexed_values[1:]),
    )
    if len(common_index) < MINIMUM_PHASE_REFLECTIONS:
        msg = (
            "dataset and reference must share at least "
            f"{MINIMUM_PHASE_REFLECTIONS} finite phase reflections"
        )
        raise PhaseAlignmentInputError(msg)

    moving_phases = moving_phase.loc[common_index].to_numpy(dtype=np.float64)
    reference_phases = reference_phase.loc[common_index].to_numpy(dtype=np.float64)
    if weighting == "uniform":
        weights = np.ones(len(common_index), dtype=np.float64)
    else:
        moving_values = moving_amplitude.loc[common_index].to_numpy(dtype=np.float64)
        reference_values = reference_amplitude.loc[common_index].to_numpy(
            dtype=np.float64
        )
        if isinstance(dataset.dtypes[amplitude_key], IntensityDtype):
            moving_values = np.sqrt(np.clip(moving_values, 0.0, None))
        if isinstance(reference.dtypes[reference_amplitude_key], IntensityDtype):
            reference_values = np.sqrt(np.clip(reference_values, 0.0, None))
        weights = np.abs(moving_values * reference_values)
    if moving_fom is not None and reference_fom is not None:
        moving_fom_values = moving_fom.loc[common_index].to_numpy(dtype=np.float64)
        reference_fom_values = reference_fom.loc[common_index].to_numpy(
            dtype=np.float64
        )
        if (
            np.any(moving_fom_values < 0.0)
            or np.any(moving_fom_values > 1.0)
            or np.any(reference_fom_values < 0.0)
            or np.any(reference_fom_values > 1.0)
        ):
            msg = "FOM values must lie between zero and one"
            raise PhaseAlignmentInputError(msg)
        weights *= moving_fom_values * reference_fom_values
    if not np.any(weights > 0.0):
        msg = "at least one phase-alignment weight must be positive"
        raise PhaseAlignmentInputError(msg)
    normalized_weights = weights / np.sum(weights)
    miller_indices = np.asarray(common_index.tolist(), dtype=np.int64)
    return (
        miller_indices,
        np.asarray(moving_phases, dtype=np.float64),
        np.asarray(reference_phases, dtype=np.float64),
        np.asarray(normalized_weights, dtype=np.float64),
    )


def _origin_shift_candidates(
    miller_indices: IntegerArray,
    phases: FloatArray,
    reference_phases: FloatArray,
    normalized_weights: FloatArray,
    spacegroup: gemmi.SpaceGroup,
    *,
    maximum_refinement_starts: int,
    search_hand: bool,
) -> tuple[OriginShiftCandidate, ...]:
    rotation_constraints = _rotation_constraints(spacegroup)
    orthonormal_polar_basis = _polar_basis(rotation_constraints)
    integer_polar_basis = _integer_polar_basis(rotation_constraints)
    allowed_grid_origins = _allowed_grid_origins(spacegroup, rotation_constraints)
    origin_cosets = _origin_cosets(
        allowed_grid_origins,
        orthonormal_polar_basis,
        spacegroup,
    )
    candidates: list[OriginShiftCandidate] = []
    hand_options = (
        (False, True)
        if search_hand and not spacegroup.is_centrosymmetric()
        else (False,)
    )
    for inverted_hand in hand_options:
        moving_phases = -phases if inverted_hand else phases
        phase_differences = np.deg2rad(
            canonicalize_phases(moving_phases - reference_phases)
        )
        internal_candidates = _rank_internal_translations(
            origin_cosets,
            integer_polar_basis,
            miller_indices,
            phase_differences,
            normalized_weights,
            spacegroup,
            maximum_refinement_starts=maximum_refinement_starts,
        )
        candidates.extend(
            OriginShiftCandidate(
                origin_shift=_phenix_origin_shift(translation),
                correlation=correlation,
                inverted_hand=inverted_hand,
            )
            for translation, correlation in internal_candidates
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: candidate.correlation,
            reverse=True,
        )
    )


def _apply_origin_shift(
    dataset: DataSet,
    origin_shift: tuple[float, float, float],
    *,
    inverted_hand: bool = False,
) -> DataSet:
    shifted = dataset.copy()
    phase_shift_degrees = (
        FULL_ROTATION_DEGREES
        * shifted.get_hkls()
        @ np.asarray(origin_shift, dtype=np.float64)
    )
    phase_sign = -1.0 if inverted_hand else 1.0
    for key in shifted.get_phase_keys():
        values = canonicalize_phases(
            phase_sign * shifted[key].to_numpy(dtype=np.float64) - phase_shift_degrees
        )
        shifted[key] = DataSeries(
            values, index=shifted.index, dtype=shifted.dtypes[key]
        )
    complex_multiplier = np.exp(-1j * np.deg2rad(phase_shift_degrees))
    for key in shifted.get_complex_keys():
        values = shifted[key].to_numpy()
        if inverted_hand:
            values = np.conjugate(values)
        shifted[key] = (values * complex_multiplier).astype(
            shifted.dtypes[key], copy=False
        )
    return shifted


def align_phases(
    dataset: DataSet,
    reference: DataSet,
    *,
    phase_key: str,
    reference_phase_key: str,
    amplitude_key: str,
    reference_amplitude_key: str,
    fom_key: Optional[str] = None,
    reference_fom_key: Optional[str] = None,
    weighting: WeightingMode = "amplitude",
    search_hand: bool = False,
    maximum_refinement_starts: int = DEFAULT_MAXIMUM_REFINEMENT_STARTS,
    max_obliquity: float = DEFAULT_MAXIMUM_OBLIQUITY,
) -> PhaseAlignmentResult:
    """Align a merged dataset by reindexing, hand, and allowed origin shift.

    Parameters
    ----------
    dataset : DataSet
        Merged moving dataset.
    reference : DataSet
        Merged reference dataset defining the target indexing and origin.
    phase_key, reference_phase_key : str
        Moving and reference phase columns in degrees.
    amplitude_key, reference_amplitude_key : str
        Moving and reference amplitude or intensity columns.
    fom_key, reference_fom_key : str, optional
        Optional Weight-typed figures of merit multiplied into the phase weights.
    weighting : {"amplitude", "uniform"}, optional
        Use amplitude-product or uniform phase weights.
    search_hand : bool, optional
        Also test the complex-conjugated moving phases. The default is ``False``.
        Centrosymmetric groups have no distinct hand and are searched only once.
    maximum_refinement_starts : int, optional
        Maximum number of FFT-local maxima refined per allowed origin coset.
    max_obliquity : float, optional
        Maximum obliquity in degrees for reindexing candidates.

    Returns
    -------
    PhaseAlignmentResult
        An aligned copy and ranked candidate diagnostics.

    Raises
    ------
    PhaseAlignmentInputError
        If the inputs are invalid or the space group has no origin ambiguity.

    Notes
    -----
    The returned origin shift uses the Phenix convention. Aligned phases satisfy
    ``phi_aligned = phi_moving - 360 * H @ origin_shift``.
    """
    _validate_dataset(dataset, data_key=amplitude_key, name="dataset")
    _validate_dataset(reference, data_key=reference_amplitude_key, name="reference")
    if not dataset.is_isomorphous(reference):
        msg = "dataset and reference must be isomorphous"
        raise PhaseAlignmentInputError(msg)
    _validate_phase_key(dataset, phase_key=phase_key, name="dataset")
    _validate_phase_key(
        reference,
        phase_key=reference_phase_key,
        name="reference",
    )
    _validate_fom_keys(
        dataset,
        reference,
        fom_key=fom_key,
        reference_fom_key=reference_fom_key,
    )
    if weighting not in ("amplitude", "uniform"):
        msg = f"weighting must be 'amplitude' or 'uniform'; got {weighting!r}"
        raise PhaseAlignmentInputError(msg)
    if not isinstance(search_hand, bool):
        msg = f"search_hand must be a bool; got {type(search_hand).__name__}"
        raise PhaseAlignmentInputError(msg)
    if not has_origin_shift_ambiguity(dataset.spacegroup):
        msg = f"space group {dataset.spacegroup.xhm()} has no origin-shift ambiguity"
        raise PhaseAlignmentInputError(msg)

    if has_reindexing_ambiguity(dataset, max_obliquity=max_obliquity):
        reindexing = reindex_by_correlation(
            dataset,
            reference,
            data_key=amplitude_key,
            reference_key=reference_amplitude_key,
            max_obliquity=max_obliquity,
        )
        phase_dataset = reindexing.dataset
    else:
        reindexing = None
        phase_dataset = _as_asu(dataset, gemmi.Op(IDENTITY_OPERATION))

    miller_indices, phases, reference_phases, normalized_weights = _matched_phase_data(
        phase_dataset,
        reference,
        phase_key=phase_key,
        reference_phase_key=reference_phase_key,
        amplitude_key=amplitude_key,
        reference_amplitude_key=reference_amplitude_key,
        fom_key=fom_key,
        reference_fom_key=reference_fom_key,
        weighting=weighting,
    )
    candidates = _origin_shift_candidates(
        miller_indices,
        phases,
        reference_phases,
        normalized_weights,
        reference.spacegroup,
        maximum_refinement_starts=maximum_refinement_starts,
        search_hand=search_hand,
    )
    best = candidates[0]
    runner_up_correlation = candidates[1].correlation if len(candidates) > 1 else None
    correlation_gap = (
        best.correlation - runner_up_correlation
        if runner_up_correlation is not None
        else None
    )
    aligned_dataset = _apply_origin_shift(
        phase_dataset, best.origin_shift, inverted_hand=best.inverted_hand
    )
    return PhaseAlignmentResult(
        dataset=aligned_dataset,
        origin_shift=best.origin_shift,
        inverted_hand=best.inverted_hand,
        correlation=best.correlation,
        runner_up_correlation=runner_up_correlation,
        correlation_gap=correlation_gap,
        candidates=candidates,
        reindexing=reindexing,
    )
