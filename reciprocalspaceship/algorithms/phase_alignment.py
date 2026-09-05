"""constrained crystallographic phase alignment"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Final, Union

import gemmi
import numpy as np
from numpy.typing import NDArray

from reciprocalspaceship.algorithms._errors import (
    PhaseAlignmentInputError,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


FloatArray: TypeAlias = NDArray[np.float64]


IntegerArray: TypeAlias = NDArray[np.int64]


SpaceGroupLike: TypeAlias = Union[str, int, gemmi.SpaceGroup]


NUMBER_OF_CRYSTALLOGRAPHIC_AXES: Final[int] = 3


ORIGIN_DENOMINATOR: Final[int] = int(gemmi.Op.DEN)


SINGULAR_VALUE_TOLERANCE: Final[float] = 1e-10


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
