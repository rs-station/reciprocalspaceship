from io import StringIO

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import reciprocalspaceship as rs


def test_against_previous_result():
    csv = """
,x,y,w
0,0.9051822979780747,0.2700784720702334,0.937803097548697
1,0.7653655064893381,0.9096902688106824,0.9022413175501339
2,0.6701438010457531,0.7643360435701648,0.8943602840313861
3,0.1524086489285047,0.9854887367590378,0.23724145891681891
4,0.8673578229262408,0.1660901563869679,0.6802234818049551
5,0.04749197327200072,0.4056733186535064,0.41411735570989516
6,0.5555482004198411,0.4273894191186294,0.36358917098272747
7,0.5463417645646479,0.5092920447904933,0.29441863366197596
8,0.31353494110452584,0.7666249814241163,0.7493823577932279
9,0.3923683608283065,0.18587807020463565,0.9318927399856036
    """
    df = pd.read_csv(StringIO(csv))
    x, y, w = df.x.to_numpy(), df.y.to_numpy(), df.w.to_numpy()
    expected_r = -0.1478766135438829

    r = rs.utils.stats.weighted_pearsonr(x, y, w)
    assert np.isclose(r, expected_r)


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
    w = np.ones(n) * 42.0
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
