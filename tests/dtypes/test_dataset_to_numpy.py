import numpy as np
import pandas as pd
import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize("dtype", [None, np.int32, np.float32, np.float64, object])
@pytest.mark.parametrize("hasna", [True, False])
def test_to_numpy_mtzints(dtype, hasna):
    """
    Test DataSet.to_numpy() with only int32-backed MTZDtypes
    """
    ds = rs.DataSet(
        {
            "X": np.random.randint(0, 100, size=100),
            "Y": np.random.randint(0, 100, size=100),
        },
        dtype="MTZInt",
    )

    if hasna:
        ds.loc[0, "X"] = np.nan

    # Can't represent array with np.int32
    if hasna and dtype is np.int32:
        with pytest.raises(ValueError):
            arr = ds.to_numpy(dtype=dtype)
    else:
        arr = ds.to_numpy(dtype=dtype)

        if dtype is None:
            # Fall back to float32
            if hasna:
                assert arr.dtype == np.float32
            # Fall back to int32
            else:
                assert arr.dtype == np.int32
        else:
            assert arr.dtype == dtype


@pytest.mark.parametrize("dtype", [None, np.float32, np.float64, object])
@pytest.mark.parametrize("hasna", [True, False])
def test_to_numpy_mtzfloats(dtype, hasna):
    """
    Test DataSet.to_numpy() with only float32-backed MTZDtypes
    """
    ds = rs.DataSet(
        {
            "X": np.random.random(100),
            "Y": np.random.random(100),
        },
        dtype="MTZReal",
    )

    if hasna:
        ds.loc[0, "X"] = np.nan

    arr = ds.to_numpy(dtype=dtype)

    if dtype is None:
        # Fall back to float32
        assert arr.dtype == np.float32
    else:
        assert arr.dtype == dtype


@pytest.mark.parametrize("dtype", [None, np.float32, np.float64, object])
@pytest.mark.parametrize("hasna", [True, False])
def test_to_numpy_mixed_mtzdtypes(dtype, hasna):
    """
    Test DataSet.to_numpy() with both int32-backed and float-32-backed MTZDtypes
    """
    ds = rs.DataSet(
        {
            "X": np.random.random(100),
            "Y": np.random.randint(100),
        },
    )
    ds["X"] = ds["X"].astype("MTZReal")
    ds["Y"] = ds["Y"].astype("MTZInt")

    if hasna:
        ds.loc[0, "X"] = np.nan

    arr = ds.to_numpy(dtype=dtype)

    if dtype is None:
        # Fall back to float32
        assert arr.dtype == np.float32
    else:
        assert arr.dtype == dtype


@pytest.mark.parametrize("non_mtzdtype", ["string", np.int32, np.float32, bool, object])
def test_to_numpy_non_mtzdtype(non_mtzdtype):
    """
    With a non-MTZDtype, DataSet.to_numpy() should always output the same thing as
    pandas.DataFrame.to_numpy(). Currently, that is "object".
    """
    ds = rs.DataSet(
        {
            "X": np.random.random(100),
            "Y": np.random.randint(100),
        },
    )
    ds["X"] = ds["X"].astype("MTZReal")
    ds["Y"] = ds["Y"].astype("MTZInt")

    ds["non_mtz"] = 0
    ds["non_mtz"] = ds["non_mtz"].astype(non_mtzdtype)

    result = ds.to_numpy()
    expected = pd.DataFrame(ds).to_numpy()

    assert result.dtype == expected.dtype
