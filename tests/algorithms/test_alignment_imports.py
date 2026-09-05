from __future__ import annotations

import importlib
from typing import Callable, get_type_hints

import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize(
    "public_object",
    [
        rs.algorithms.has_reindexing_ambiguity,
        rs.algorithms.reindex_by_correlation,
        rs.algorithms.ReindexingResult,
        rs.algorithms.has_origin_shift_ambiguity,
    ],
)
def test_public_alignment_annotations_resolve(
    public_object: Callable[..., object],
) -> None:
    # Regression: importing DataSet during package initialization must not shadow
    # the concat function.
    dataset_module = importlib.import_module("reciprocalspaceship.dataset")

    assert dataset_module.concat is rs.concat
    assert get_type_hints(public_object)
