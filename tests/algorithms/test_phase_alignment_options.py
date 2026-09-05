from __future__ import annotations

from typing import Final

import gemmi
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

import reciprocalspaceship as rs
from reciprocalspaceship.algorithms._errors import PhaseAlignmentInputError
from reciprocalspaceship.algorithms.phase_alignment import _matched_phase_data

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]

SPACEGROUP: Final[gemmi.SpaceGroup] = gemmi.SpaceGroup("P 1")
UNIT_CELL: Final[gemmi.UnitCell] = gemmi.UnitCell(
    20.0,
    25.0,
    31.0,
    71.0,
    83.0,
    97.0,
)
FULL_ROTATION_DEGREES: Final[float] = 360.0


def _dataset(
    miller_indices: IntegerArray,
    amplitudes: FloatArray,
    phases: FloatArray,
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
    dataset["F"] = dataset["F"].astype("SFAmplitude")
    dataset["PHI"] = dataset["PHI"].astype("Phase")
    return dataset


def _add_fom(dataset: rs.DataSet, values: FloatArray, *, key: str = "FOM") -> None:
    dataset[key] = values
    dataset[key] = dataset[key].astype("Weight")


def _align(
    moving: rs.DataSet,
    reference: rs.DataSet,
    *,
    search_hand: bool = False,
) -> rs.algorithms.PhaseAlignmentResult:
    return rs.algorithms.align_phases(
        moving,
        reference,
        phase_key="PHI",
        reference_phase_key="PHI",
        amplitude_key="F",
        reference_amplitude_key="F",
        search_hand=search_hand,
    )


def test_fom_values_multiply_amplitude_weights() -> None:
    miller_indices = np.asarray(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        dtype=np.int64,
    )
    amplitudes = np.asarray((1.0, 2.0, 4.0), dtype=np.float64)
    phases = np.zeros(3, dtype=np.float64)
    moving = _dataset(miller_indices, amplitudes, phases)
    reference = _dataset(miller_indices, amplitudes, phases)
    moving_fom = np.asarray((0.2, 0.5, 0.8), dtype=np.float64)
    reference_fom = np.asarray((0.3, 0.7, 0.9), dtype=np.float64)
    _add_fom(moving, moving_fom)
    _add_fom(reference, reference_fom)

    *_, weights = _matched_phase_data(
        moving,
        reference,
        phase_key="PHI",
        reference_phase_key="PHI",
        amplitude_key="F",
        reference_amplitude_key="F",
        fom_key="FOM",
        reference_fom_key="FOM",
        weighting="amplitude",
    )

    expected = amplitudes**2 * moving_fom * reference_fom
    expected /= np.sum(expected)
    np.testing.assert_allclose(weights, expected)


@pytest.mark.parametrize("invalid_fom", [-0.1, 1.1])
def test_fom_values_must_be_probabilities(invalid_fom: float) -> None:
    miller_indices = np.asarray(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        dtype=np.int64,
    )
    dataset = _dataset(
        miller_indices,
        np.ones(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )
    _add_fom(dataset, np.asarray((0.2, 0.5, invalid_fom), dtype=np.float64))

    with pytest.raises(PhaseAlignmentInputError, match="between zero and one"):
        rs.algorithms.align_phases(
            dataset,
            dataset,
            phase_key="PHI",
            reference_phase_key="PHI",
            amplitude_key="F",
            reference_amplitude_key="F",
            fom_key="FOM",
            reference_fom_key="FOM",
        )


def test_fom_keys_must_be_supplied_as_a_pair() -> None:
    miller_indices = np.asarray(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        dtype=np.int64,
    )
    dataset = _dataset(
        miller_indices,
        np.ones(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )
    _add_fom(dataset, np.ones(3, dtype=np.float64))

    with pytest.raises(PhaseAlignmentInputError, match="both be set"):
        rs.algorithms.align_phases(
            dataset,
            dataset,
            phase_key="PHI",
            reference_phase_key="PHI",
            amplitude_key="F",
            reference_amplitude_key="F",
            fom_key="FOM",
        )


@pytest.mark.parametrize("complex_dtype", ["complex64", "complex128"])
def test_hand_search_and_complex_columns_are_opt_in(complex_dtype: str) -> None:
    random_number_generator = np.random.default_rng(seed=20260822)
    miller_indices = random_number_generator.integers(low=-3, high=4, size=(80, 3))
    miller_indices = np.unique(miller_indices, axis=0)
    miller_indices = miller_indices[np.any(miller_indices != 0, axis=1)]
    miller_indices = miller_indices[rs.utils.in_asu(miller_indices, SPACEGROUP)]
    amplitudes = random_number_generator.lognormal(
        mean=2.0,
        sigma=0.5,
        size=len(miller_indices),
    )
    reference_phases = random_number_generator.uniform(
        low=-180.0,
        high=180.0,
        size=len(miller_indices),
    )
    origin_shift = np.asarray((0.137, -0.271, 0.419), dtype=np.float64)
    moving_phases = rs.utils.canonicalize_phases(
        -reference_phases - FULL_ROTATION_DEGREES * miller_indices @ origin_shift,
    )
    reference = _dataset(miller_indices, amplitudes, reference_phases)
    moving = _dataset(miller_indices, amplitudes, moving_phases)
    reference["FC"] = amplitudes * np.exp(1j * np.deg2rad(reference_phases))
    moving["FC"] = amplitudes * np.exp(1j * np.deg2rad(moving_phases))
    moving["FC"] = moving["FC"].astype(complex_dtype)
    moving_before = moving.copy()

    without_hand = _align(moving, reference)
    with_hand = _align(moving, reference, search_hand=True)

    assert without_hand.inverted_hand is False
    assert with_hand.inverted_hand is True
    # Regression: applying the origin used to promote complex64 columns to complex128.
    assert with_hand.dataset["FC"].dtype == moving["FC"].dtype
    pd.testing.assert_frame_equal(moving, moving_before)
    np.testing.assert_allclose(with_hand.origin_shift, origin_shift, atol=1e-6)
    np.testing.assert_allclose(with_hand.correlation, 1.0, atol=1e-10)
    np.testing.assert_allclose(
        with_hand.dataset["FC"].to_numpy(),
        reference["FC"].to_numpy(),
        atol=2e-5,
    )


def test_centrosymmetric_hand_search_does_not_duplicate_solutions() -> None:
    # Regression: conjugation counted the same centrosymmetric solution twice,
    # so default confidence gates rejected even a perfect alignment.
    miller_indices = np.asarray(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (1, 2, 3)), dtype=np.int64
    )
    reference = _dataset(miller_indices, np.ones(5), np.zeros(5))
    reference.spacegroup = "P -1"

    result = rs.algorithms.align_phases(
        reference,
        reference,
        phase_key="PHI",
        reference_phase_key="PHI",
        amplitude_key="F",
        reference_amplitude_key="F",
        search_hand=True,
    )

    assert not result.inverted_hand
    assert all(not candidate.inverted_hand for candidate in result.candidates)
    np.testing.assert_allclose(result.origin_shift, 0.0, atol=1e-12)
    assert np.isclose(result.correlation, 1.0)
