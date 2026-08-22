from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

import gemmi
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

import reciprocalspaceship as rs

FloatArray = NDArray[np.float64]


IntegerArray = NDArray[np.int64]


SPACEGROUP: Final[gemmi.SpaceGroup] = gemmi.SpaceGroup("P 21 21 2")


UNIT_CELL: Final[gemmi.UnitCell] = gemmi.UnitCell(30.0, 30.0, 50.0, 90.0, 90.0, 90.0)


REINDEXING_OPERATION: Final[gemmi.Op] = gemmi.Op("-y,-x,-z")


DATA_KEY: Final[str] = "F_MOVING"


REFERENCE_KEY: Final[str] = "F_REFERENCE"


@dataclass(frozen=True)
class ReindexingPair:
    """Synthetic datasets related by a known indexing operation."""

    moving: rs.DataSet
    reference: rs.DataSet
    operation: gemmi.Op


def _asu_miller_indices(
    spacegroup: gemmi.SpaceGroup = SPACEGROUP,
) -> IntegerArray:
    miller_indices = np.mgrid[-5:6, -5:6, -3:4].reshape(3, -1).T
    nonzero = np.any(miller_indices != 0, axis=1)
    in_asu = rs.utils.in_asu(miller_indices, spacegroup)
    present = ~rs.utils.is_absent(miller_indices, spacegroup)
    return np.asarray(miller_indices[nonzero & in_asu & present], dtype=np.int64)


def _make_dataset(
    miller_indices: IntegerArray,
    values: FloatArray,
    *,
    key: str,
    dtype: str = "SFAmplitude",
    spacegroup: Optional[gemmi.SpaceGroup] = SPACEGROUP,
    cell: Optional[gemmi.UnitCell] = UNIT_CELL,
    merged: Optional[bool] = True,
) -> rs.DataSet:
    dataset = rs.DataSet(
        {
            "H": miller_indices[:, 0],
            "K": miller_indices[:, 1],
            "L": miller_indices[:, 2],
            key: values,
        },
        spacegroup=spacegroup,
        cell=cell,
        merged=merged,
    )
    dataset.set_index(["H", "K", "L"], inplace=True)
    dataset[key] = dataset[key].astype(dtype)
    return dataset


def _make_reindexing_pair(
    *,
    values: Optional[FloatArray] = None,
    dtype: str = "SFAmplitude",
) -> ReindexingPair:
    miller_indices = _asu_miller_indices()
    if values is None:
        random_number_generator = np.random.default_rng(seed=20260822)
        values = random_number_generator.lognormal(
            mean=3.0,
            sigma=0.7,
            size=len(miller_indices),
        )
    reference = _make_dataset(
        miller_indices,
        values,
        key=REFERENCE_KEY,
        dtype=dtype,
    )
    moving_in_reference_indexing = _make_dataset(
        miller_indices,
        values,
        key=DATA_KEY,
        dtype=dtype,
    )
    moving = moving_in_reference_indexing.apply_symop(
        REINDEXING_OPERATION.inverse()
    ).hkl_to_asu()
    moving.drop(columns="M/ISYM", inplace=True)
    return ReindexingPair(moving, reference, REINDEXING_OPERATION)


@pytest.fixture(scope="session")
def reindexing_pair() -> ReindexingPair:
    return _make_reindexing_pair()


def test_reindex_by_correlation_smoke_and_does_not_mutate_inputs(
    reindexing_pair: ReindexingPair,
) -> None:
    moving_before = reindexing_pair.moving.copy()
    reference_before = reindexing_pair.reference.copy()

    result = rs.algorithms.reindex_by_correlation(
        reindexing_pair.moving,
        reindexing_pair.reference,
        data_key=DATA_KEY,
        reference_key=REFERENCE_KEY,
    )

    assert isinstance(result, rs.algorithms.ReindexingResult)
    assert isinstance(result.dataset, rs.DataSet)
    assert isinstance(result.operation, gemmi.Op)
    assert isinstance(result.correlation, float)
    assert isinstance(result.runner_up_correlation, float)
    assert isinstance(result.correlation_gap, float)
    assert isinstance(result.candidates, tuple)
    assert result.candidates[0].operation == result.operation
    assert np.isclose(result.candidates[0].correlation, result.correlation)
    assert result.dataset is not reindexing_pair.moving
    assert result.dataset.spacegroup.xhm() == SPACEGROUP.xhm()
    np.testing.assert_allclose(result.dataset.cell.parameters, UNIT_CELL.parameters)
    assert result.dataset.merged is True
    pd.testing.assert_frame_equal(reindexing_pair.moving, moving_before)
    pd.testing.assert_frame_equal(reindexing_pair.reference, reference_before)


def test_reindex_by_correlation_selects_correct_operation(
    reindexing_pair: ReindexingPair,
) -> None:
    # Regression: alternate indexing must be corrected before origin alignment.
    result = rs.algorithms.reindex_by_correlation(
        reindexing_pair.moving,
        reindexing_pair.reference,
        data_key=DATA_KEY,
        reference_key=REFERENCE_KEY,
    )

    assert result.operation == reindexing_pair.operation
    np.testing.assert_allclose(result.correlation, 1.0, atol=1e-6)
    common_indices = result.dataset.index.intersection(reindexing_pair.reference.index)
    np.testing.assert_allclose(
        result.dataset.loc[common_indices, DATA_KEY].to_numpy(dtype=np.float64),
        reindexing_pair.reference.loc[common_indices, REFERENCE_KEY].to_numpy(
            dtype=np.float64
        ),
        rtol=1e-6,
    )


def test_reindex_by_correlation_reports_operation_applied_to_moving_data() -> None:
    spacegroup = gemmi.SpaceGroup("P 1")
    cubic_cell = gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)
    expected_operation = gemmi.Op("y,z,x")
    candidate_indices = np.mgrid[-4:5, -4:5, -4:5].reshape(3, -1).T
    miller_indices = np.asarray(
        candidate_indices[
            np.any(candidate_indices != 0, axis=1)
            & rs.utils.in_asu(candidate_indices, spacegroup)
        ],
        dtype=np.int64,
    )
    random_number_generator = np.random.default_rng(seed=20260822)
    values = random_number_generator.lognormal(
        mean=3.0,
        sigma=0.7,
        size=len(miller_indices),
    )
    reference = _make_dataset(
        miller_indices,
        values,
        key=REFERENCE_KEY,
        spacegroup=spacegroup,
        cell=cubic_cell,
    )
    moving_in_reference_indexing = _make_dataset(
        miller_indices,
        values,
        key=DATA_KEY,
        spacegroup=spacegroup,
        cell=cubic_cell,
    )
    moving = moving_in_reference_indexing.apply_symop(
        expected_operation.inverse(),
    ).hkl_to_asu()
    moving.drop(columns="M/ISYM", inplace=True)

    # Regression: the reported non-involutive operation must map moving to reference.
    result = rs.algorithms.reindex_by_correlation(
        moving,
        reference,
        data_key=DATA_KEY,
        reference_key=REFERENCE_KEY,
    )

    assert result.operation.triplet() == expected_operation.triplet()
    np.testing.assert_allclose(result.correlation, 1.0, atol=1e-10)


def test_reindex_by_correlation_can_select_identity() -> None:
    reindexing_pair = _make_reindexing_pair()
    moving = reindexing_pair.reference.rename(columns={REFERENCE_KEY: DATA_KEY})

    result = rs.algorithms.reindex_by_correlation(
        moving,
        reindexing_pair.reference,
        data_key=DATA_KEY,
        reference_key=REFERENCE_KEY,
    )

    assert result.operation.triplet() == "x,y,z"
    np.testing.assert_allclose(result.correlation, 1.0, atol=1e-6)


def test_reindexing_preserves_extra_columns_and_row_order(
    reindexing_pair: ReindexingPair,
) -> None:
    # Regression: scoring only HKLs/data must still transform all output columns.
    moving = reindexing_pair.moving.copy()
    moving["row_number"] = np.arange(len(moving))
    moving["PHI"] = rs.DataSeries(
        np.linspace(-170.0, 170.0, len(moving)), dtype="P"
    ).to_numpy()
    moving["PHI"] = moving["PHI"].astype("P")
    moving["FC"] = moving.to_structurefactor(DATA_KEY, "PHI")
    expected = moving.apply_symop(reindexing_pair.operation).hkl_to_asu()
    expected.drop(columns="M/ISYM", inplace=True)
    expected["FC"] = expected["FC"].astype(moving["FC"].dtype)

    result = rs.algorithms.reindex_by_correlation(
        moving,
        reindexing_pair.reference,
        data_key=DATA_KEY,
        reference_key=REFERENCE_KEY,
    )

    pd.testing.assert_frame_equal(result.dataset, expected)


def test_reindex_by_correlation_rejects_spacegroup_without_ambiguity() -> None:
    miller_indices = _asu_miller_indices(gemmi.SpaceGroup("P 21 21 21"))
    values = np.arange(1.0, len(miller_indices) + 1.0)
    nonambiguous_cell = gemmi.UnitCell(30.0, 37.0, 50.0, 90.0, 90.0, 90.0)
    moving = _make_dataset(
        miller_indices,
        values,
        key=DATA_KEY,
        spacegroup=gemmi.SpaceGroup("P 21 21 21"),
        cell=nonambiguous_cell,
    )
    reference = _make_dataset(
        miller_indices,
        values,
        key=REFERENCE_KEY,
        spacegroup=gemmi.SpaceGroup("P 21 21 21"),
        cell=nonambiguous_cell,
    )

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="reindex"):
        rs.algorithms.reindex_by_correlation(
            moving,
            reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


def test_reindex_by_correlation_matches_common_hkls() -> None:
    reindexing_pair = _make_reindexing_pair()
    aligned_moving = reindexing_pair.moving.apply_symop(
        reindexing_pair.operation
    ).hkl_to_asu()
    aligned_moving.drop(columns="M/ISYM", inplace=True)
    aligned_moving = aligned_moving.iloc[7:].sample(frac=1.0, random_state=20260822)
    moving = aligned_moving.apply_symop(
        reindexing_pair.operation.inverse()
    ).hkl_to_asu()
    moving.drop(columns="M/ISYM", inplace=True)
    reference = reindexing_pair.reference.iloc[:-11].sample(
        frac=1.0,
        random_state=20260823,
    )
    expected_common_indices = aligned_moving.index.intersection(reference.index)

    result = rs.algorithms.reindex_by_correlation(
        moving,
        reference,
        data_key=DATA_KEY,
        reference_key=REFERENCE_KEY,
    )

    assert result.operation == reindexing_pair.operation
    # All operations must be scored on one shared reflection set to avoid overlap bias.
    candidate_reflection_counts = {
        candidate.number_of_reflections for candidate in result.candidates
    }
    assert len(candidate_reflection_counts) == 1
    assert result.candidates[0].number_of_reflections <= len(expected_common_indices)
    np.testing.assert_allclose(result.correlation, 1.0, atol=1e-6)


@pytest.mark.parametrize("dtype", ["SFAmplitude", "Intensity"])
def test_reindex_by_correlation_accepts_amplitudes_and_intensities(dtype: str) -> None:
    reindexing_pair = _make_reindexing_pair(dtype=dtype)

    result = rs.algorithms.reindex_by_correlation(
        reindexing_pair.moving,
        reindexing_pair.reference,
        data_key=DATA_KEY,
        reference_key=REFERENCE_KEY,
    )

    assert result.operation == reindexing_pair.operation
    np.testing.assert_allclose(result.correlation, 1.0, atol=1e-6)
