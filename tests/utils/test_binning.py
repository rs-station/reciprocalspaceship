import pytest
import numpy as np
import reciprocalspaceship as rs

@pytest.mark.parametrize("bins", [10, 20, 50])
@pytest.mark.parametrize("ascending", [True, False])
def test_bin_by_percentile(bins, ascending):
    data = np.linspace(0, 100, 1000)
    assignments, labels = rs.utils.bin_by_percentile(data, bins, ascending)

    # Make sure number of bins is correct
    assert len(labels) == bins
    assert len(assignments) == len(data)
    assert len(np.unique(assignments)) == bins
    
    if ascending:
        np.all(np.diff(assignments) >= 0)
    else:
        np.all(np.diff(assignments) <= 0)
