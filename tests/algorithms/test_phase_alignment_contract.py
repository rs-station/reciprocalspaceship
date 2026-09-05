from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

import gemmi
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

import reciprocalspaceship as rs

PHASE_DATA_DIRECTORY: Final[Path] = Path(__file__).parents[1] / "data" / "fmodel"


FULL_ROTATION_DEGREES: Final[float] = 360.0


PERMISSIVE_CORRELATION: Final[float] = -1.0


PERMISSIVE_CORRELATION_GAP: Final[float] = 0.0


FloatArray = NDArray[np.float64]


IntegerArray = NDArray[np.int64]


def _shift_phases(
    reference: rs.DataSet,
    origin_shift: tuple[float, float, float],
    *,
    phase_key: str = "PHIFMODEL",
) -> rs.DataSet:
    moving = reference.copy()
    phase_shifts = (
        FULL_ROTATION_DEGREES
        * moving.get_hkls()
        @ np.asarray(origin_shift, dtype=np.float64)
    )
    moving[phase_key] = rs.utils.canonicalize_phases(
        moving[phase_key].to_numpy(dtype=np.float64) + phase_shifts
    ).astype(np.float32)
    moving[phase_key] = moving[phase_key].astype("Phase")
    return moving


def _synthetic_dataset(
    miller_indices: IntegerArray,
    amplitudes: FloatArray,
    phases: FloatArray,
    *,
    spacegroup: gemmi.SpaceGroup,
    cell: gemmi.UnitCell,
) -> rs.DataSet:
    dataset = rs.DataSet(
        {
            "H": miller_indices[:, 0],
            "K": miller_indices[:, 1],
            "L": miller_indices[:, 2],
            "F": amplitudes,
            "PHI": phases,
        },
        spacegroup=spacegroup,
        cell=cell,
        merged=True,
    )
    dataset.set_index(["H", "K", "L"], inplace=True)
    dataset["F"] = dataset["F"].astype("SFAmplitude")
    dataset["PHI"] = dataset["PHI"].astype("Phase")
    return dataset


def _align(
    moving: rs.DataSet,
    reference: rs.DataSet,
    *,
    phase_key: str = "PHIFMODEL",
    reference_phase_key: str = "PHIFMODEL",
    amplitude_key: str = "FMODEL",
    reference_amplitude_key: str = "FMODEL",
    weighting: Literal["amplitude", "uniform"] = "amplitude",
) -> rs.algorithms.PhaseAlignmentResult:
    return rs.algorithms.align_phases(
        moving,
        reference,
        phase_key=phase_key,
        reference_phase_key=reference_phase_key,
        amplitude_key=amplitude_key,
        reference_amplitude_key=reference_amplitude_key,
        weighting=weighting,
        warning_correlation=PERMISSIVE_CORRELATION,
        minimum_correlation=PERMISSIVE_CORRELATION,
        minimum_correlation_gap=PERMISSIVE_CORRELATION_GAP,
    )


def test_align_phases_oblique_rhombohedral_polar_axis() -> None:
    random_number_generator = np.random.default_rng(seed=20260814)
    spacegroup = gemmi.SpaceGroup("R 3:R")
    cell = gemmi.UnitCell(50.0, 50.0, 50.0, 75.0, 75.0, 75.0)
    candidate_indices = np.mgrid[-8:9, -8:9, -8:9].reshape(3, -1).T
    present = ~rs.utils.is_absent(candidate_indices, spacegroup)
    in_asu = rs.utils.in_asu(candidate_indices, spacegroup)
    miller_indices = np.asarray(
        candidate_indices[present & in_asu][:800],
        dtype=np.int64,
    )
    amplitudes = random_number_generator.lognormal(
        mean=3.0,
        sigma=0.7,
        size=len(miller_indices),
    )
    reference_phases = random_number_generator.uniform(
        low=-180.0,
        high=180.0,
        size=len(miller_indices),
    )
    expected_origin_shift = (-0.137, -0.137, -0.137)
    reference = _synthetic_dataset(
        miller_indices,
        amplitudes,
        reference_phases,
        spacegroup=spacegroup,
        cell=cell,
    )
    moving = _shift_phases(reference, expected_origin_shift, phase_key="PHI")

    result = _align(
        moving,
        reference,
        phase_key="PHI",
        reference_phase_key="PHI",
        amplitude_key="F",
        reference_amplitude_key="F",
    )

    np.testing.assert_allclose(
        result.origin_shift,
        expected_origin_shift,
        atol=1e-6,
    )


def test_align_phases_snaps_discrete_origin_with_noise() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "9LYZ.mtz"))
    expected_origin_shift = (-0.5, -0.5, -0.5)
    moving = _shift_phases(reference, expected_origin_shift)
    random_number_generator = np.random.default_rng(seed=20260814)
    moving["PHIFMODEL"] += random_number_generator.normal(
        loc=0.0,
        scale=10.0,
        size=len(moving),
    )

    result = _align(moving, reference)

    np.testing.assert_allclose(
        result.origin_shift,
        expected_origin_shift,
        atol=1e-12,
    )


def test_align_phases_rejects_unidentifiable_continuous_origin() -> None:
    miller_indices = np.asarray(((1, 0, 0), (2, 0, 0), (3, 0, 0)), dtype=np.int64)
    amplitudes = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)
    phases = np.asarray((10.0, 20.0, 30.0), dtype=np.float64)
    spacegroup = gemmi.SpaceGroup("P 1")
    cell = gemmi.UnitCell(41.0, 53.0, 67.0, 79.0, 83.0, 74.0)
    reference = _synthetic_dataset(
        miller_indices,
        amplitudes,
        phases,
        spacegroup=spacegroup,
        cell=cell,
    )

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="identify"):
        _align(
            reference.copy(),
            reference,
            phase_key="PHI",
            reference_phase_key="PHI",
            amplitude_key="F",
            reference_amplitude_key="F",
        )


def test_align_phases_raises_when_every_refinement_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_minimize(*_args: object, **_kwargs: object) -> OptimizeResult:
        return OptimizeResult(
            success=False,
            fun=np.inf,
            jac=np.ones(3, dtype=np.float64),
            message="forced failure",
        )

    monkeypatch.setattr(
        "reciprocalspaceship.algorithms.phase_alignment.minimize",
        failed_minimize,
    )
    miller_indices = np.asarray(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.int64)
    amplitudes = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)
    phases = np.asarray((10.0, 20.0, 30.0), dtype=np.float64)
    spacegroup = gemmi.SpaceGroup("P 1")
    cell = gemmi.UnitCell(41.0, 53.0, 67.0, 79.0, 83.0, 74.0)
    reference = _synthetic_dataset(
        miller_indices,
        amplitudes,
        phases,
        spacegroup=spacegroup,
        cell=cell,
    )

    with pytest.raises(rs.algorithms.PhaseAlignmentOptimizationError, match="forced"):
        _align(
            reference.copy(),
            reference,
            phase_key="PHI",
            reference_phase_key="PHI",
            amplitude_key="F",
            reference_amplitude_key="F",
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing-phase", "MISSING"),
        ("wrong-phase-dtype", "Phase"),
        ("invalid-weighting", "weighting"),
        ("one-fom", "both"),
        ("non-bool-hand", "search_hand"),
    ],
)
def test_align_phases_validates_dataset_interface(change: str, message: str) -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    moving = reference.copy()
    kwargs: dict[str, object] = {
        "phase_key": "PHIFMODEL",
        "reference_phase_key": "PHIFMODEL",
        "amplitude_key": "FMODEL",
        "reference_amplitude_key": "FMODEL",
    }
    if change == "missing-phase":
        kwargs["phase_key"] = "MISSING"
    elif change == "wrong-phase-dtype":
        moving["PHIFMODEL"] = moving["PHIFMODEL"].astype("MTZReal")
    elif change == "invalid-weighting":
        kwargs["weighting"] = "invalid"
    elif change == "one-fom":
        moving["FOM"] = np.ones(len(moving), dtype=np.float32)
        kwargs["fom_key"] = "FOM"
    else:
        kwargs["search_hand"] = 1

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match=message):
        rs.algorithms.align_phases(moving, reference, **kwargs)


def test_align_phases_rejects_spacegroup_without_origin_ambiguity() -> None:
    spacegroup = gemmi.SpaceGroup(199)
    cell = gemmi.UnitCell(50.0, 50.0, 50.0, 90.0, 90.0, 90.0)
    candidate_indices = np.mgrid[-3:4, -3:4, -3:4].reshape(3, -1).T
    present = ~rs.utils.is_absent(candidate_indices, spacegroup)
    in_asu = rs.utils.in_asu(candidate_indices, spacegroup)
    miller_indices = np.asarray(candidate_indices[present & in_asu], dtype=np.int64)
    amplitudes = np.arange(1.0, len(miller_indices) + 1.0)
    phases = np.zeros(len(miller_indices), dtype=np.float64)
    reference = _synthetic_dataset(
        miller_indices,
        amplitudes,
        phases,
        spacegroup=spacegroup,
        cell=cell,
    )

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="no origin"):
        _align(
            reference.copy(),
            reference,
            phase_key="PHI",
            reference_phase_key="PHI",
            amplitude_key="F",
            reference_amplitude_key="F",
        )
