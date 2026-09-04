from __future__ import annotations

from typing import Final, Union, cast

import gemmi
import numpy as np
import pytest

import reciprocalspaceship as rs

SpaceGroupLike = Union[str, int, gemmi.SpaceGroup]

UNIQUE_ORIGIN_SPACEGROUP_NUMBERS: Final[tuple[int, ...]] = (
    197,
    199,
    204,
    206,
    211,
    214,
    217,
    220,
    229,
    230,
)
SPACEGROUP_SETTINGS: Final[tuple[gemmi.SpaceGroup, ...]] = tuple(
    gemmi.spacegroup_table()
)


def _dataset_with_symmetry(
    spacegroup: SpaceGroupLike,
    cell_parameters: tuple[float, float, float, float, float, float],
) -> rs.DataSet:
    return rs.DataSet(
        spacegroup=spacegroup,
        cell=gemmi.UnitCell(*cell_parameters),
    )


def test_has_reindexing_ambiguity_smoke() -> None:
    dataset = _dataset_with_symmetry(
        "P 21 21 2",
        (10.0, 10.0, 20.0, 90.0, 90.0, 90.0),
    )

    result = rs.algorithms.has_reindexing_ambiguity(dataset)

    assert result is True


def test_has_reindexing_ambiguity_requires_dataset() -> None:
    with pytest.raises(ValueError, match="rs.DataSet"):
        rs.algorithms.has_reindexing_ambiguity("P 1")


def test_has_reindexing_ambiguity_depends_on_unit_cell() -> None:
    square_ab_dataset = _dataset_with_symmetry(
        "P 21 21 2",
        (10.0, 10.0, 20.0, 90.0, 90.0, 90.0),
    )
    rectangular_ab_dataset = _dataset_with_symmetry(
        "P 21 21 2",
        (10.0, 11.0, 20.0, 90.0, 90.0, 90.0),
    )

    assert rs.algorithms.has_reindexing_ambiguity(square_ab_dataset)
    assert not rs.algorithms.has_reindexing_ambiguity(rectangular_ab_dataset)


def test_has_reindexing_ambiguity_respects_maximum_obliquity() -> None:
    pseudo_merohedral_dataset = _dataset_with_symmetry(
        "P 21 21 2",
        (10.0, 10.1, 20.0, 90.0, 90.0, 90.0),
    )

    assert not rs.algorithms.has_reindexing_ambiguity(
        pseudo_merohedral_dataset,
        max_obliquity=0.1,
    )
    assert rs.algorithms.has_reindexing_ambiguity(
        pseudo_merohedral_dataset,
        max_obliquity=1.0,
    )


@pytest.mark.parametrize(
    "cell_parameters",
    [
        (np.nan, 10.0, 20.0, 90.0, 90.0, 90.0),
        (-10.0, 10.0, 20.0, 90.0, 90.0, 90.0),
        (10.0, 10.0, 20.0, -90.0, 90.0, 90.0),
        (10.0, 10.0, 20.0, 10.0, 10.0, 150.0),
    ],
)
def test_has_reindexing_ambiguity_rejects_invalid_cell_geometry(
    cell_parameters: tuple[float, float, float, float, float, float],
) -> None:
    # Regression: malformed cells reached Gemmi's lattice search unchecked.
    dataset = _dataset_with_symmetry("P 21 21 2", cell_parameters)

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="valid geometry"):
        rs.algorithms.has_reindexing_ambiguity(dataset)


def test_has_reindexing_ambiguity_rejects_overflowed_obliquity() -> None:
    # Regression: converting a large integer leaked an undocumented OverflowError.
    dataset = _dataset_with_symmetry("P 21 21 2", (10, 10, 20, 90, 90, 90))

    with pytest.raises(ValueError, match="max_obliquity"):
        rs.algorithms.has_reindexing_ambiguity(dataset, max_obliquity=10**1000)


@pytest.mark.parametrize(
    ("missing_attribute", "dataset"),
    [
        (
            "spacegroup",
            rs.DataSet(cell=gemmi.UnitCell(10.0, 11.0, 20.0, 90.0, 90.0, 90.0)),
        ),
        ("cell", rs.DataSet(spacegroup="P 21 21 2")),
    ],
)
def test_has_reindexing_ambiguity_requires_symmetry_metadata(
    missing_attribute: str,
    dataset: rs.DataSet,
) -> None:
    with pytest.raises(ValueError, match=missing_attribute):
        rs.algorithms.has_reindexing_ambiguity(dataset)


@pytest.mark.parametrize("max_obliquity", [-1.0, np.nan, np.inf])
def test_has_reindexing_ambiguity_rejects_invalid_maximum_obliquity(
    max_obliquity: float,
) -> None:
    dataset = _dataset_with_symmetry(
        "P 21 21 2",
        (10.0, 10.0, 20.0, 90.0, 90.0, 90.0),
    )

    with pytest.raises(ValueError, match="max_obliquity"):
        rs.algorithms.has_reindexing_ambiguity(
            dataset,
            max_obliquity=max_obliquity,
        )


@pytest.mark.parametrize("spacegroup", [1, "P 1", gemmi.SpaceGroup(1)])
def test_has_origin_shift_ambiguity_accepts_spacegroup_types(
    spacegroup: SpaceGroupLike,
) -> None:
    assert rs.algorithms.has_origin_shift_ambiguity(spacegroup) is True


@pytest.mark.parametrize(
    "invalid_spacegroup",
    [None, False, 0, 231, "not a space group", 999, 10**100],
)
def test_has_origin_shift_ambiguity_normalizes_invalid_inputs(
    invalid_spacegroup: object,
) -> None:
    with pytest.raises(
        rs.algorithms.PhaseAlignmentInputError,
        match="spacegroup could not be converted",
    ):
        rs.algorithms.has_origin_shift_ambiguity(
            cast(SpaceGroupLike, invalid_spacegroup)
        )


@pytest.mark.parametrize(
    "spacegroup",
    ["P 21 21 21", "P 61", "R 3:R"],
    ids=["discrete", "polar", "oblique-polar"],
)
def test_has_origin_shift_ambiguity_representative_cases(spacegroup: str) -> None:
    assert rs.algorithms.has_origin_shift_ambiguity(spacegroup) is True


@pytest.mark.parametrize("spacegroup_number", UNIQUE_ORIGIN_SPACEGROUP_NUMBERS)
def test_has_origin_shift_ambiguity_quotients_centering_translations(
    spacegroup_number: int,
) -> None:
    # Regression: body-centering alone does not create a distinguishable origin.
    assert rs.algorithms.has_origin_shift_ambiguity(spacegroup_number) is False


@pytest.mark.parametrize("spacegroup_number", range(1, 231))
def test_has_origin_shift_ambiguity_all_reference_spacegroups(
    spacegroup_number: int,
) -> None:
    # Regression: these values were independently generated from CCTBX seminvariants.
    expected = spacegroup_number not in UNIQUE_ORIGIN_SPACEGROUP_NUMBERS

    assert rs.algorithms.has_origin_shift_ambiguity(spacegroup_number) is expected


@pytest.mark.parametrize("spacegroup", SPACEGROUP_SETTINGS)
def test_has_origin_shift_ambiguity_is_setting_invariant(
    spacegroup: gemmi.SpaceGroup,
) -> None:
    expected = spacegroup.number not in UNIQUE_ORIGIN_SPACEGROUP_NUMBERS

    assert rs.algorithms.has_origin_shift_ambiguity(spacegroup) is expected
