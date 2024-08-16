from inspect import signature

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import reciprocalspaceship as rs


def test_reset_index_dataseries():
    """
    Minimal example from GH#223
    """
    result = rs.DataSeries(range(10)).reset_index()
    expected = pd.Series(range(10)).reset_index()
    expected = rs.DataSet(expected)
    assert_frame_equal(result, expected)


def test_reset_index_signature(dataset_hkl):
    """
    Test call signature of rs.DataSet.reset_index() matches call signature of
    pd.DataFrame.reset_index() using default parameters
    """
    df = pd.DataFrame(dataset_hkl)
    sig = signature(pd.DataFrame.reset_index)
    bsig = sig.bind(df)
    bsig.apply_defaults()

    expected = df.reset_index(*bsig.args[1:], **bsig.kwargs)
    result = dataset_hkl.reset_index(*bsig.args[1:], **bsig.kwargs)
    result = pd.DataFrame(result)

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("names", ["H", "K", ["H", "K"]])
def test_set_index_signature(dataset_hkl, names):
    """
    Test call signature of rs.DataSet.set_index() matches call signature of
    pd.DataFrame.set_index() using default parameters
    """
    ds = dataset_hkl.reset_index()
    df = pd.DataFrame(ds)
    sig = signature(pd.DataFrame.set_index)
    bsig = sig.bind(df, names)
    bsig.apply_defaults()

    expected = df.set_index(*bsig.args[1:], **bsig.kwargs)
    result = ds.set_index(*bsig.args[1:], **bsig.kwargs)
    result = pd.DataFrame(result)

    assert_frame_equal(result, expected)
