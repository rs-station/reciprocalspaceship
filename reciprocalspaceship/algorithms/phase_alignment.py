"""constrained crystallographic phase alignment"""

from __future__ import annotations

from heapq import heappush, heapreplace
from itertools import product
from math import gcd, prod
from typing import TYPE_CHECKING, Final, Union

import gemmi
import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len
from scipy.ndimage import maximum_filter

from reciprocalspaceship.algorithms._errors import (
    PhaseAlignmentInputError,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


FloatArray: TypeAlias = NDArray[np.float64]


IntegerArray: TypeAlias = NDArray[np.int64]


ComplexArray: TypeAlias = NDArray[np.complex128]


SpaceGroupLike: TypeAlias = Union[str, int, gemmi.SpaceGroup]


NUMBER_OF_CRYSTALLOGRAPHIC_AXES: Final[int] = 3


FULL_ROTATION_RADIANS: Final[float] = float(2.0 * np.pi)


ORIGIN_DENOMINATOR: Final[int] = int(gemmi.Op.DEN)


SINGULAR_VALUE_TOLERANCE: Final[float] = 1e-10


MAXIMUM_FFT_MEMORY_BYTES: Final[int] = 1_073_741_824


FFT_BYTES_PER_GRID_POINT: Final[int] = 80


FFT_SLAB_BYTES_PER_GRID_POINT: Final[int] = 80


DEFAULT_MAXIMUM_REFINEMENT_STARTS: Final[int] = 4_096


LOCAL_MAXIMUM_NEIGHBORHOOD: Final[int] = 3


POLAR_AXIS_CONSTRAINT_RANK: Final[int] = NUMBER_OF_CRYSTALLOGRAPHIC_AXES - 1


PERIODIC_OFFSETS: Final[FloatArray] = np.asarray(
    tuple(product((-1.0, 0.0, 1.0), repeat=NUMBER_OF_CRYSTALLOGRAPHIC_AXES)),
    dtype=np.float64,
)


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
