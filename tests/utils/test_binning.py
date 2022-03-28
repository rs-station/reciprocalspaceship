import numpy as np
import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize("data", [np.arange(0, 11)])
@pytest.mark.parametrize("bin_edges", [[0, 5, 3, 10], [10, 3, 5, 0]])
@pytest.mark.parametrize("right_inclusive", [True, False])
def test_assign_with_binedges_nonmonotonic(data, bin_edges, right_inclusive):
    """
    Test assign_with_binedges() raises ValueError with non-monotonic bin edges
    """
    with pytest.raises(ValueError):
        rs.utils.assign_with_binedges(data, bin_edges, right_inclusive)


@pytest.mark.parametrize("data", [np.arange(0, 11)])
@pytest.mark.parametrize("bin_edges", [[1, 3, 5, 10], [10, 5, 1]])
@pytest.mark.parametrize("right_inclusive", [True, False])
def test_assign_with_binedges_dataoutside(data, bin_edges, right_inclusive):
    """
    Test assign_with_binedges() raises ValueError if `bin_edges` do not fully
    contain `data`
    """
    with pytest.raises(ValueError):
        rs.utils.assign_with_binedges(data, bin_edges, right_inclusive)


@pytest.mark.parametrize(
    "data",
    [
        np.linspace(0, 100, 1000),  # Simulates merged data
        np.repeat(np.linspace(0, 100, 1000), 3),  # Simulates unmerged data
    ],
)
@pytest.mark.parametrize("bins", [10, 20, 50])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
def test_bin_by_percentile(data, bins, ascending, shuffle):
    # Order shouldn't matter
    if shuffle:
        np.random.shuffle(data)

    assignments, labels = rs.utils.bin_by_percentile(data, bins, ascending)

    # Make sure number of bins is correct
    assert len(labels) == bins
    assert len(assignments) == len(data)
    assert len(np.unique(assignments)) == bins

    if ascending:
        np.all(np.diff(assignments) >= 0)
    else:
        np.all(np.diff(assignments) <= 0)
