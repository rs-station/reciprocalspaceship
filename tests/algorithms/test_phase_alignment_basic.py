from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import gemmi
import numpy as np
import pytest

import reciprocalspaceship as rs

PHASE_DATA_DIRECTORY: Final[Path] = Path(__file__).parents[1] / "data" / "fmodel"
FULL_ROTATION_DEGREES: Final[float] = 360.0


def _shift_phases(
    reference: rs.DataSet,
    origin_shift: tuple[float, float, float],
) -> rs.DataSet:
    moving = reference.copy()
    phase_shifts = (
        FULL_ROTATION_DEGREES
        * moving.get_hkls()
        @ np.asarray(origin_shift, dtype=np.float64)
    )
    moving["PHIFMODEL"] = rs.utils.canonicalize_phases(
        moving["PHIFMODEL"].to_numpy(dtype=np.float64) + phase_shifts,
    ).astype(np.float32)
    moving["PHIFMODEL"] = moving["PHIFMODEL"].astype("Phase")
    return moving


def _align(
    moving: rs.DataSet, reference: rs.DataSet
) -> rs.algorithms.PhaseAlignmentResult:
    return rs.algorithms.align_phases(
        moving,
        reference,
        phase_key="PHIFMODEL",
        reference_phase_key="PHIFMODEL",
        amplitude_key="FMODEL",
        reference_amplitude_key="FMODEL",
    )


@pytest.mark.parametrize(
    ("filename", "expected_origin_shift"),
    [
        ("9LYZ.mtz", (-0.5, -0.5, -0.5)),
        ("3KXE.mtz", (-0.5, 0.0, -0.5)),
        ("6OVT.mtz", (0.0, 0.0, -0.137)),
    ],
)
def test_align_phases_real_spacegroup_cases(
    filename: str,
    expected_origin_shift: tuple[float, float, float],
) -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / filename))
    moving = _shift_phases(reference, expected_origin_shift)

    result = _align(moving, reference)

    assert isinstance(result, rs.algorithms.PhaseAlignmentResult)
    assert result.inverted_hand is False
    np.testing.assert_allclose(result.origin_shift, expected_origin_shift, atol=1e-6)
    np.testing.assert_allclose(result.correlation, 1.0, atol=1e-10)


def test_align_phases_reindexes_before_origin_search() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    reindexing_operation = reference.reindexing_ops[0]
    moving = reference.apply_symop(reindexing_operation)

    # Regression: alternate P61 indexing must be corrected before fitting the origin.
    result = _align(moving, reference)

    assert result.reindexing is not None
    assert result.reindexing.operation.triplet() == reindexing_operation.triplet()
    np.testing.assert_allclose(result.correlation, 1.0, atol=1e-10)
    np.testing.assert_allclose(result.origin_shift, 0.0, atol=1e-6)


def test_align_phases_returns_a_copy_with_phenix_sign() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    expected_origin_shift = (0.0, 0.0, -0.137)
    moving = _shift_phases(reference, expected_origin_shift)
    original_phases = moving["PHIFMODEL"].copy()

    result = _align(moving, reference)

    np.testing.assert_allclose(result.origin_shift, expected_origin_shift, atol=1e-6)
    np.testing.assert_allclose(
        rs.utils.canonicalize_phases(
            result.dataset["PHIFMODEL"].to_numpy(dtype=np.float64)
            - reference["PHIFMODEL"].to_numpy(dtype=np.float64),
        ),
        0.0,
        atol=2e-5,
    )
    np.testing.assert_allclose(moving["PHIFMODEL"], original_phases, atol=1e-12)


@pytest.mark.parametrize("invalid_value", [0, 1, True, 1.5])
def test_align_phases_rejects_invalid_maximum_refinement_starts(
    invalid_value: object,
) -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))

    # Regression: one retained maximum can hide a tied runner-up.
    with pytest.raises(
        rs.algorithms.PhaseAlignmentInputError,
        match="maximum_refinement_starts",
    ):
        rs.algorithms.align_phases(
            reference,
            reference,
            phase_key="PHIFMODEL",
            reference_phase_key="PHIFMODEL",
            amplitude_key="FMODEL",
            reference_amplitude_key="FMODEL",
            maximum_refinement_starts=cast(int, invalid_value),
        )


def test_align_phases_warns_for_low_but_usable_correlation() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    moving = _shift_phases(reference, (0.0, 0.0, -0.137))
    noise = np.random.default_rng(seed=20260822).normal(
        loc=0.0,
        scale=40.0,
        size=len(moving),
    )
    moving["PHIFMODEL"] += noise

    with pytest.warns(rs.algorithms.LowCorrelationWarning, match="origin shift"):
        rs.algorithms.align_phases(
            moving,
            reference,
            phase_key="PHIFMODEL",
            reference_phase_key="PHIFMODEL",
            amplitude_key="FMODEL",
            reference_amplitude_key="FMODEL",
            warning_correlation=0.99,
            minimum_correlation=-1.0,
            minimum_correlation_gap=0.0,
        )


def test_align_phases_rejects_an_unclear_solution() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    moving = _shift_phases(reference, (0.0, 0.0, -0.137))
    noise = np.random.default_rng(seed=20260823).normal(
        loc=0.0,
        scale=40.0,
        size=len(moving),
    )
    moving["PHIFMODEL"] += noise

    with pytest.raises(rs.algorithms.NoClearSolutionError, match="best correlation"):
        rs.algorithms.align_phases(
            moving,
            reference,
            phase_key="PHIFMODEL",
            reference_phase_key="PHIFMODEL",
            amplitude_key="FMODEL",
            reference_amplitude_key="FMODEL",
            warning_correlation=0.99,
            minimum_correlation=0.99,
            minimum_correlation_gap=0.0,
        )


def test_align_phases_rejects_nonisomorphous_inputs() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    moving = reference.copy()
    moving.cell = gemmi.UnitCell(20.0, 21.0, 22.0, 90.0, 90.0, 90.0)

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="isomorphous"):
        _align(moving, reference)
