from __future__ import annotations

from typing import Final

import gemmi
import numpy as np
import pytest
from numpy.typing import NDArray

import reciprocalspaceship as rs
from reciprocalspaceship.algorithms._errors import PhaseAlignmentInputError
from reciprocalspaceship.algorithms.phase_alignment import (
    _apply_origin_shift,
    _matched_phase_data,
    _validate_phase_key,
)

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]

SPACEGROUP: Final[gemmi.SpaceGroup] = gemmi.SpaceGroup("P 1")
UNIT_CELL: Final[gemmi.UnitCell] = gemmi.UnitCell(
    20.0,
    25.0,
    30.0,
    90.0,
    90.0,
    90.0,
)
FULL_ROTATION_DEGREES: Final[float] = 360.0


def _dataset(
    miller_indices: IntegerArray,
    amplitudes: FloatArray,
    phases: FloatArray,
    *,
    amplitude_dtype: str = "SFAmplitude",
) -> rs.DataSet:
    dataset = rs.DataSet(
        {
            "H": miller_indices[:, 0],
            "K": miller_indices[:, 1],
            "L": miller_indices[:, 2],
            "F": amplitudes,
            "PHI": phases,
        },
        spacegroup=SPACEGROUP,
        cell=UNIT_CELL,
        merged=True,
    )
    dataset.set_index(["H", "K", "L"], inplace=True)
    dataset["F"] = dataset["F"].astype(amplitude_dtype)
    dataset["PHI"] = dataset["PHI"].astype("Phase")
    return dataset


def _matched(
    moving: rs.DataSet,
    reference: rs.DataSet,
) -> tuple[IntegerArray, FloatArray, FloatArray, FloatArray]:
    return _matched_phase_data(
        moving,
        reference,
        phase_key="PHI",
        reference_phase_key="PHI",
        amplitude_key="F",
        reference_amplitude_key="F",
        fom_key=None,
        reference_fom_key=None,
        weighting="amplitude",
    )


def test_matched_phase_data_aligns_rows_by_miller_index() -> None:
    miller_indices = np.asarray(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)),
        dtype=np.int64,
    )
    amplitudes = np.asarray((1.0, 2.0, 3.0, 4.0), dtype=np.float64)
    phases = np.asarray((10.0, 20.0, 30.0, 40.0), dtype=np.float64)
    reference = _dataset(miller_indices, amplitudes, phases)
    moving = _dataset(miller_indices[::-1], amplitudes[::-1], phases[::-1])

    matched_hkls, moving_phases, reference_phases, _ = _matched(moving, reference)

    np.testing.assert_array_equal(matched_hkls, miller_indices[::-1])
    np.testing.assert_allclose(moving_phases, phases[::-1])
    np.testing.assert_allclose(reference_phases, phases[::-1])


def test_matched_phase_data_uses_amplitude_product_weights() -> None:
    miller_indices = np.asarray(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        dtype=np.int64,
    )
    moving_amplitudes = np.asarray((1.0, 2.0, 4.0), dtype=np.float64)
    reference_amplitudes = np.asarray((3.0, 5.0, 7.0), dtype=np.float64)
    phases = np.zeros(3, dtype=np.float64)
    moving = _dataset(miller_indices, moving_amplitudes, phases)
    reference = _dataset(miller_indices, reference_amplitudes, phases)

    _, _, _, normalized_weights = _matched(moving, reference)

    expected = moving_amplitudes * reference_amplitudes
    expected /= np.sum(expected)
    np.testing.assert_allclose(normalized_weights, expected)


def test_intensity_weights_match_amplitude_weights() -> None:
    miller_indices = np.asarray(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        dtype=np.int64,
    )
    amplitudes = np.asarray((1.0, 2.0, 4.0), dtype=np.float64)
    phases = np.zeros(3, dtype=np.float64)
    amplitude_dataset = _dataset(miller_indices, amplitudes, phases)
    intensity_dataset = _dataset(
        miller_indices,
        amplitudes**2,
        phases,
        amplitude_dtype="Intensity",
    )

    amplitude_weights = _matched(amplitude_dataset, amplitude_dataset)[3]
    intensity_weights = _matched(intensity_dataset, intensity_dataset)[3]

    np.testing.assert_allclose(intensity_weights, amplitude_weights)


def test_matched_phase_data_requires_three_finite_reflections() -> None:
    miller_indices = np.asarray(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        dtype=np.int64,
    )
    amplitudes = np.ones(3, dtype=np.float64)
    reference = _dataset(
        miller_indices,
        amplitudes,
        np.asarray((10.0, 20.0, 30.0), dtype=np.float64),
    )
    moving = reference.copy()
    moving.iloc[2, moving.columns.get_loc("PHI")] = np.nan

    with pytest.raises(PhaseAlignmentInputError, match="at least 3"):
        _matched(moving, reference)


def test_apply_origin_shift_uses_phenix_sign_and_preserves_dtype() -> None:
    miller_indices = np.asarray(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)),
        dtype=np.int64,
    )
    reference_phases = np.asarray((10.0, 20.0, 30.0, 40.0), dtype=np.float64)
    origin_shift = np.asarray((0.137, -0.271, 0.419), dtype=np.float64)
    shifted_phases = rs.utils.canonicalize_phases(
        reference_phases + FULL_ROTATION_DEGREES * miller_indices @ origin_shift,
    )
    moving = _dataset(
        miller_indices,
        np.ones(4, dtype=np.float64),
        shifted_phases,
    )

    aligned = _apply_origin_shift(moving, tuple(origin_shift))

    assert aligned is not moving
    assert isinstance(aligned.dtypes["PHI"], rs.PhaseDtype)
    np.testing.assert_allclose(
        rs.utils.canonicalize_phases(
            aligned["PHI"].to_numpy(dtype=np.float64) - reference_phases,
        ),
        0.0,
        atol=2e-5,
    )


def test_validate_phase_key_rejects_missing_or_untyped_columns() -> None:
    miller_indices = np.asarray(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        dtype=np.int64,
    )
    dataset = _dataset(
        miller_indices,
        np.ones(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )

    with pytest.raises(PhaseAlignmentInputError, match="MISSING"):
        _validate_phase_key(dataset, phase_key="MISSING", name="dataset")

    dataset["PHI"] = dataset["PHI"].astype(float)
    with pytest.raises(PhaseAlignmentInputError, match="Phase MTZ dtype"):
        _validate_phase_key(dataset, phase_key="PHI", name="dataset")
