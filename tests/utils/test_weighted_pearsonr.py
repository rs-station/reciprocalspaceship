import numpy as np
import pytest
from scipy.stats import pearsonr

import reciprocalspaceship as rs




def test_weighted_pearsonr():
    n = 100

    x, y, sigx, sigy = np.random.random((4, n))
    w = np.sqrt(sigx * sigx + sigy * sigy)

    # Test execution
    rs.utils.stats.weighted_pearsonr(x, y, w)

    # Test against scipy pearsonr
    w = np.ones(n)
    r = rs.utils.stats.weighted_pearsonr(x, y, w)
    expected_r = pearsonr(x, y)[0]
    assert np.isclose(r, expected_r)

    # Test against scipy with another uniform weight value
    w = np.ones(n) * 42.
    r = rs.utils.stats.weighted_pearsonr(x, y, w)
    expected_r = pearsonr(x, y)[0]
    assert np.isclose(r, expected_r)


def test_weighted_pearsonr_batched():
    # Test batch execution
    a, b, n = 2, 3, 100
    x, y, sigx, sigy = np.random.random((4, a, b, n))
    w = np.sqrt(sigx * sigx + sigy * sigy)
    r = rs.utils.stats.weighted_pearsonr(x, y, w)
    assert np.all(np.array(r.shape) == np.array([a, b]))

    # Test against scipy pearsonr
    w = np.ones((a, b, n))
    r = rs.utils.stats.weighted_pearsonr(x, y, w)
    for i in range(a):
        for j in range(b):
            expected_r = pearsonr(x[i, j], y[i, j])[0]
            assert np.isclose(r[i, j], expected_r)
