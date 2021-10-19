import numpy as np
import pytest

import reciprocalspaceship as rs


def test_merge_valueerror(hewl_merged):
    """
    Confirm rs.algorithms.merge() raises ValueError when invoked with
    merged DataSet
    """
    with pytest.raises(ValueError):
        merged = rs.algorithms.merge(hewl_merged)


@pytest.mark.parametrize(
    "keys",
    [
        None,
        ["I", "SIGI"],
        ["I", "SigI"],
    ],
)
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("anomalous", [False, True])
def test_merge(hewl_unmerged, hewl_merged, keys, sort, anomalous):
    """Test rs.algorithms.merge() against AIMLESS output"""
    if keys is None:
        merged = rs.algorithms.merge(hewl_unmerged, sort=sort, anomalous=anomalous)
    elif not (keys[0] in hewl_unmerged.columns and keys[1] in hewl_unmerged.columns):
        with pytest.raises(KeyError):
            merged = rs.algorithms.merge(
                hewl_unmerged, keys[0], keys[1], sort=sort, anomalous=anomalous
            )
        return
    else:
        merged = rs.algorithms.merge(
            hewl_unmerged, keys[0], keys[1], sort=sort, anomalous=anomalous
        )

    # Check DataSet attributes
    assert merged.merged
    assert merged.spacegroup.xhm() == hewl_merged.spacegroup.xhm()
    assert merged.cell.a == hewl_merged.cell.a

    if sort:
        assert merged.index.is_monotonic_increasing

    # Note: AIMLESS zero-fills empty observations, whereas we use NaNs
    if not anomalous:
        hewl_merged["I"] = hewl_merged["IMEAN"]
        hewl_merged["SIGI"] = hewl_merged["SIGIMEAN"]
        hewl_merged["N"] = rs.DataSeries(
            (hewl_merged["N(+)"] + hewl_merged["N(-)"])
            / (hewl_merged.label_centrics().CENTRIC + 1),
            dtype="I",
        )

    for key in merged.columns:
        assert np.allclose(
            merged.loc[hewl_merged.index, key].fillna(0),
            hewl_merged.loc[hewl_merged.index, key],
        )
