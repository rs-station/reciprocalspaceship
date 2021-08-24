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
def test_merge(hewl_unmerged, hewl_merged, keys, sort):
    """Test rs.algorithms.merge() against AIMLESS output"""
    if keys is None:
        merged = rs.algorithms.merge(hewl_unmerged, sort=sort)
    elif not (keys[0] in hewl_unmerged.columns and keys[1] in hewl_unmerged.columns):
        with pytest.raises(KeyError):
            merged = rs.algorithms.merge(hewl_unmerged, keys[0], keys[1], sort=sort)
        return
    else:
        merged = rs.algorithms.merge(hewl_unmerged, keys[0], keys[1], sort=sort)

    # Check DataSet attributes
    assert merged.merged
    assert merged.spacegroup.xhm() == hewl_merged.spacegroup.xhm()
    assert merged.cell.a == hewl_merged.cell.a
    assert merged.index.is_monotonic_increasing == sort

    # Note: AIMLESS zero-fills empty observations, whereas we use NaNs
    for key in merged.columns:
        assert np.allclose(
            merged.loc[hewl_merged.index, key].fillna(0),
            hewl_merged.loc[hewl_merged.index, key],
        )
