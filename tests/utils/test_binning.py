import numpy as np
import pytest

import reciprocalspaceship as rs


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
