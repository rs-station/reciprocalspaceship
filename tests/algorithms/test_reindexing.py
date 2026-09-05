from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional, cast

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
    moving["PHI"] = np.linspace(-170.0, 170.0, len(moving))
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


def test_reindex_by_correlation_raises_when_candidates_are_tied() -> None:
    miller_indices = _asu_miller_indices()
    values = np.asarray(
        5.0
        + miller_indices[:, 0] ** 2
        + miller_indices[:, 1] ** 2
        + 2.0 * miller_indices[:, 2] ** 2,
        dtype=np.float64,
    )
    reindexing_pair = _make_reindexing_pair(values=values)

    with pytest.raises(rs.algorithms.NoClearSolutionError, match="gap|ambiguous|clear"):
        rs.algorithms.reindex_by_correlation(
            reindexing_pair.moving,
            reindexing_pair.reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
            minimum_correlation=0.0,
            minimum_correlation_gap=0.01,
        )


def test_reindex_by_correlation_warns_for_low_accepted_correlation() -> None:
    reindexing_pair = _make_reindexing_pair()
    random_number_generator = np.random.default_rng(seed=20260823)
    noisy_moving = reindexing_pair.moving.copy()
    noise = random_number_generator.normal(
        loc=0.0,
        scale=float(noisy_moving[DATA_KEY].std()),
        size=len(noisy_moving),
    )
    noisy_moving[DATA_KEY] = np.maximum(
        noisy_moving[DATA_KEY].to_numpy(dtype=np.float64) + noise,
        0.01,
    ).astype(np.float32)
    noisy_moving[DATA_KEY] = noisy_moving[DATA_KEY].astype("SFAmplitude")

    with pytest.warns(
        rs.algorithms.LowCorrelationWarning,
        match="correlation",
    ) as warning_records:
        result = rs.algorithms.reindex_by_correlation(
            noisy_moving,
            reindexing_pair.reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
            minimum_correlation=-1.0,
            minimum_correlation_gap=0.0,
            warning_correlation=0.95,
        )

    assert result.correlation < 0.95
    assert warning_records[0].filename == __file__


def test_reindex_by_correlation_raises_for_unacceptable_correlation() -> None:
    reindexing_pair = _make_reindexing_pair()
    random_number_generator = np.random.default_rng(seed=20260824)
    unrelated_moving = reindexing_pair.moving.copy()
    unrelated_moving[DATA_KEY] = random_number_generator.lognormal(
        mean=3.0,
        sigma=0.7,
        size=len(unrelated_moving),
    ).astype(np.float32)
    unrelated_moving[DATA_KEY] = unrelated_moving[DATA_KEY].astype("SFAmplitude")

    with pytest.raises(rs.algorithms.NoClearSolutionError, match="correlation|clear"):
        rs.algorithms.reindex_by_correlation(
            unrelated_moving,
            reindexing_pair.reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
            minimum_correlation=0.99,
            minimum_correlation_gap=0.0,
        )


@pytest.mark.parametrize("target", ["moving", "reference"])
@pytest.mark.parametrize("merged", [False, None])
def test_reindex_by_correlation_requires_merged_datasets(
    target: str,
    merged: Optional[bool],
) -> None:
    reindexing_pair = _make_reindexing_pair()
    moving = reindexing_pair.moving.copy()
    reference = reindexing_pair.reference.copy()
    if target == "moving":
        moving.merged = merged
    else:
        reference.merged = merged

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="merged"):
        rs.algorithms.reindex_by_correlation(
            moving,
            reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


@pytest.mark.parametrize("target", ["moving", "reference"])
def test_reindex_by_correlation_requires_unique_hkls(target: str) -> None:
    reindexing_pair = _make_reindexing_pair()
    moving = reindexing_pair.moving.copy()
    reference = reindexing_pair.reference.copy()
    source = moving if target == "moving" else reference
    key = DATA_KEY if target == "moving" else REFERENCE_KEY
    duplicate_hkls = np.vstack((source.get_hkls(), source.get_hkls()[0]))
    duplicate_values = np.append(
        source[key].to_numpy(dtype=np.float64),
        float(source[key].iloc[0]),
    )
    duplicated = _make_dataset(duplicate_hkls, duplicate_values, key=key)
    if target == "moving":
        moving = duplicated
    else:
        reference = duplicated

    with pytest.raises(
        rs.algorithms.PhaseAlignmentInputError, match="unique|duplicate"
    ):
        rs.algorithms.reindex_by_correlation(
            moving,
            reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


@pytest.mark.parametrize("target", ["moving", "reference"])
@pytest.mark.parametrize("metadata", ["spacegroup", "cell"])
def test_reindex_by_correlation_requires_crystallographic_metadata(
    target: str,
    metadata: str,
) -> None:
    reindexing_pair = _make_reindexing_pair()
    moving = reindexing_pair.moving.copy()
    reference = reindexing_pair.reference.copy()
    source = moving if target == "moving" else reference
    key = DATA_KEY if target == "moving" else REFERENCE_KEY
    spacegroup = None if metadata == "spacegroup" else source.spacegroup
    cell = None if metadata == "cell" else source.cell
    dataset = _make_dataset(
        source.get_hkls(),
        source[key].to_numpy(dtype=np.float64),
        key=key,
        spacegroup=spacegroup,
        cell=cell,
    )
    if metadata == "spacegroup":
        assert dataset.spacegroup is None
    else:
        assert dataset.cell is None
    if target == "moving":
        moving = dataset
    else:
        reference = dataset

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match=metadata):
        rs.algorithms.reindex_by_correlation(
            moving,
            reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


@pytest.mark.parametrize("metadata", ["spacegroup", "cell"])
def test_reindex_by_correlation_requires_isomorphous_datasets(metadata: str) -> None:
    reindexing_pair = _make_reindexing_pair()
    reference = reindexing_pair.reference.copy()
    if metadata == "spacegroup":
        reference.spacegroup = gemmi.SpaceGroup("P 2 2 2")
    else:
        reference.cell = gemmi.UnitCell(60.0, 30.0, 50.0, 90.0, 90.0, 90.0)

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="isomorphous"):
        rs.algorithms.reindex_by_correlation(
            reindexing_pair.moving,
            reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


@pytest.mark.parametrize("target", ["moving", "reference"])
def test_reindex_by_correlation_requires_existing_data_keys(target: str) -> None:
    reindexing_pair = _make_reindexing_pair()
    data_key = "MISSING" if target == "moving" else DATA_KEY
    reference_key = "MISSING" if target == "reference" else REFERENCE_KEY

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="MISSING"):
        rs.algorithms.reindex_by_correlation(
            reindexing_pair.moving,
            reindexing_pair.reference,
            data_key=data_key,
            reference_key=reference_key,
        )


@pytest.mark.parametrize("target", ["moving", "reference"])
def test_reindex_by_correlation_requires_amplitude_or_intensity_keys(
    target: str,
) -> None:
    reindexing_pair = _make_reindexing_pair()
    moving = reindexing_pair.moving.copy()
    reference = reindexing_pair.reference.copy()
    dataset = moving if target == "moving" else reference
    key = DATA_KEY if target == "moving" else REFERENCE_KEY
    dataset[key] = dataset[key].astype("MTZReal")

    with pytest.raises(
        rs.algorithms.PhaseAlignmentInputError,
        match="amplitude|intensity|dtype",
    ):
        rs.algorithms.reindex_by_correlation(
            moving,
            reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


def test_reindex_by_correlation_rejects_anomalous_columns() -> None:
    reindexing_pair = _make_reindexing_pair()
    moving = reindexing_pair.moving.copy()
    moving["F(+)"] = moving[DATA_KEY].astype("SFAmplitude")
    moving["F(+)"] = moving["F(+)"].astype("FriedelSFAmplitude")

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="anomalous"):
        rs.algorithms.reindex_by_correlation(
            moving,
            reindexing_pair.reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


def test_reindex_by_correlation_rejects_hendrickson_lattman_columns() -> None:
    reindexing_pair = _make_reindexing_pair()
    moving = reindexing_pair.moving.copy()
    moving["HLA"] = np.ones(len(moving), dtype=np.float32)
    moving["HLA"] = moving["HLA"].astype("HendricksonLattman")

    with pytest.raises(
        rs.algorithms.PhaseAlignmentInputError,
        match="Hendrickson-Lattman",
    ):
        rs.algorithms.reindex_by_correlation(
            moving,
            reindexing_pair.reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


@pytest.mark.parametrize("target", ["dataset", "reference"])
def test_reindex_by_correlation_requires_datasets(target: str) -> None:
    reindexing_pair = _make_reindexing_pair()
    moving = reindexing_pair.moving
    reference = reindexing_pair.reference
    if target == "dataset":
        moving = cast(rs.DataSet, pd.DataFrame(reindexing_pair.moving))
    else:
        reference = cast(rs.DataSet, pd.DataFrame(reindexing_pair.reference))

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="rs.DataSet"):
        rs.algorithms.reindex_by_correlation(
            moving,
            reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


@pytest.mark.parametrize("target", ["dataset", "reference"])
def test_reindex_by_correlation_requires_hkl_columns(target: str) -> None:
    reindexing_pair = _make_reindexing_pair()
    moving = reindexing_pair.moving
    reference = reindexing_pair.reference
    source = moving if target == "dataset" else reference
    key = DATA_KEY if target == "dataset" else REFERENCE_KEY
    without_hkls = rs.DataSet(
        {key: source[key].to_numpy(dtype=np.float64)},
        spacegroup=source.spacegroup,
        cell=source.cell,
        merged=True,
    )
    without_hkls[key] = without_hkls[key].astype("SFAmplitude")
    if target == "dataset":
        moving = without_hkls
    else:
        reference = without_hkls

    with pytest.raises(
        rs.algorithms.PhaseAlignmentInputError,
        match="Miller indices|H, K, and L",
    ):
        rs.algorithms.reindex_by_correlation(
            moving,
            reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


@pytest.mark.parametrize(
    ("invalid_h", "message"),
    [
        ("not-an-index", "numeric"),
        (np.nan, "finite"),
        (1.5, "integer-valued"),
        (2**32, "int32 range"),
        (-(2**31), "int32 range"),
        (1e100, "int32 range"),
    ],
)
def test_reindex_by_correlation_validates_hkl_values(
    invalid_h: object,
    message: str,
) -> None:
    # Regression: oversized integers wrapped into apparently valid reflections.
    reindexing_pair = _make_reindexing_pair()
    table = reindexing_pair.reference.reset_index()
    h_values = table["H"].to_numpy(dtype=object)
    h_values[0] = invalid_h
    table["H"] = h_values
    invalid_reference = rs.DataSet(
        table,
        spacegroup=reindexing_pair.reference.spacegroup,
        cell=reindexing_pair.reference.cell,
        merged=True,
    ).set_index(["H", "K", "L"])

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match=message):
        rs.algorithms.reindex_by_correlation(
            reindexing_pair.moving,
            invalid_reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


@pytest.mark.parametrize(
    (
        "warning_correlation",
        "minimum_correlation",
        "minimum_correlation_gap",
        "max_obliquity",
        "message",
    ),
    [
        (np.nan, 0.2, 0.05, 1e-6, "warning_correlation"),
        (1.01, 0.2, 0.05, 1e-6, "warning_correlation"),
        (0.5, -1.01, 0.05, 1e-6, "minimum_correlation"),
        (0.5, 1.01, 0.05, 1e-6, "minimum_correlation"),
        (0.5, 0.2, -0.01, 1e-6, "minimum_correlation_gap"),
        (0.5, 0.2, 2.01, 1e-6, "minimum_correlation_gap"),
        (0.5, 0.2, 0.05, -0.01, "max_obliquity"),
        (0.5, 0.2, 0.05, np.inf, "max_obliquity"),
        (cast(float, "bad"), 0.2, 0.05, 1e-6, "warning_correlation"),
        (0.5, 0.2, cast(float, "bad"), 1e-6, "minimum_correlation_gap"),
        (0.5, 0.2, 0.05, cast(float, "bad"), "max_obliquity"),
    ],
)
def test_reindex_by_correlation_rejects_invalid_thresholds(
    warning_correlation: float,
    minimum_correlation: float,
    minimum_correlation_gap: float,
    max_obliquity: float,
    message: str,
) -> None:
    reindexing_pair = _make_reindexing_pair()

    with pytest.raises(ValueError, match=message):
        rs.algorithms.reindex_by_correlation(
            reindexing_pair.moving,
            reindexing_pair.reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
            warning_correlation=warning_correlation,
            minimum_correlation=minimum_correlation,
            minimum_correlation_gap=minimum_correlation_gap,
            max_obliquity=max_obliquity,
        )


@pytest.mark.parametrize("target", ["dataset", "reference"])
def test_reindex_by_correlation_rejects_duplicate_asu_hkls(target: str) -> None:
    reindexing_pair = _make_reindexing_pair()
    moving = reindexing_pair.moving
    reference = reindexing_pair.reference
    source = moving if target == "dataset" else reference
    key = DATA_KEY if target == "dataset" else REFERENCE_KEY
    equivalent_hkl = -source.get_hkls()[0]
    assert not np.any(np.all(source.get_hkls() == equivalent_hkl, axis=1))
    miller_indices = np.vstack((source.get_hkls(), equivalent_hkl))
    values = np.append(
        source[key].to_numpy(dtype=np.float64),
        float(source[key].iloc[0]),
    )
    duplicated_in_asu = _make_dataset(miller_indices, values, key=key)
    if target == "dataset":
        moving = duplicated_in_asu
    else:
        reference = duplicated_in_asu

    with pytest.raises(
        rs.algorithms.PhaseAlignmentInputError,
        match="unique Miller indices after mapping to the ASU",
    ):
        rs.algorithms.reindex_by_correlation(
            moving,
            reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


@pytest.mark.parametrize("target", ["dataset", "reference"])
def test_reindex_by_correlation_rejects_nonpositive_resolution_bin(target: str) -> None:
    reindexing_pair = _make_reindexing_pair(dtype="Intensity")
    moving = reindexing_pair.moving.copy()
    reference = reindexing_pair.reference.copy()
    dataset = moving if target == "dataset" else reference
    key = DATA_KEY if target == "dataset" else REFERENCE_KEY
    dataset[key] = np.full(len(dataset), -1.0, dtype=np.float32)
    dataset[key] = dataset[key].astype("Intensity")

    with pytest.raises(
        rs.algorithms.PhaseAlignmentInputError,
        match="positive finite mean",
    ):
        rs.algorithms.reindex_by_correlation(
            moving,
            reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


@pytest.mark.parametrize("target", ["dataset", "reference"])
def test_reindex_by_correlation_rejects_constant_data(target: str) -> None:
    reindexing_pair = _make_reindexing_pair()
    moving = reindexing_pair.moving.copy()
    reference = reindexing_pair.reference.copy()
    dataset = moving if target == "dataset" else reference
    key = DATA_KEY if target == "dataset" else REFERENCE_KEY
    dataset[key] = np.ones(len(dataset), dtype=np.float32)
    dataset[key] = dataset[key].astype("SFAmplitude")

    with pytest.raises(
        rs.algorithms.PhaseAlignmentInputError,
        match="undefined for constant",
    ):
        rs.algorithms.reindex_by_correlation(
            moving,
            reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )


def test_reindex_by_correlation_requires_three_common_reflections() -> None:
    miller_indices = np.asarray(((0, 1, 1), (1, 0, 1)), dtype=np.int64)
    values = np.asarray((10.0, 20.0), dtype=np.float64)
    reference = _make_dataset(miller_indices, values, key=REFERENCE_KEY)
    moving_in_reference_indexing = _make_dataset(
        miller_indices,
        values,
        key=DATA_KEY,
    )
    moving = moving_in_reference_indexing.apply_symop(
        REINDEXING_OPERATION.inverse()
    ).hkl_to_asu()
    moving.drop(columns="M/ISYM", inplace=True)

    with pytest.raises(
        rs.algorithms.PhaseAlignmentInputError,
        match="at least 3 finite reflections",
    ):
        rs.algorithms.reindex_by_correlation(
            moving,
            reference,
            data_key=DATA_KEY,
            reference_key=REFERENCE_KEY,
        )
