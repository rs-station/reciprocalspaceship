import unittest
from os.path import abspath, dirname, join

import numpy as np
import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize("fraction", [0.05, 0.10, 0.15])
@pytest.mark.parametrize("ccp4_convention", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("seed", [None, 2022])
def test_add_rfree(data_fmodel, fraction, ccp4_convention, inplace, seed):
    """
    Test rs.utils.add_rfree
    """
    data_copy = data_fmodel.copy()
    rfree = rs.utils.add_rfree(
        data_fmodel,
        fraction=fraction,
        ccp4_convention=ccp4_convention,
        inplace=inplace,
        seed=seed,
    )

    if ccp4_convention:
        label_name = "FreeR_flag"
        assert label_name in rfree.columns
        assert np.sum(rfree.loc[:, label_name] == 0) / len(rfree) < fraction + 0.1
    else:
        label_name = "R-free-flags"
        assert label_name in rfree.columns
        assert np.sum(rfree.loc[:, label_name] == 1) / len(rfree) < fraction + 0.1

    if inplace:
        assert id(data_fmodel) == id(rfree)
        assert label_name in data_fmodel.columns
        assert np.all(data_copy == rfree.loc[:, rfree.columns != label_name])
    else:
        assert id(data_fmodel) != id(rfree)
        assert label_name not in data_fmodel.columns
        assert np.all(data_fmodel == data_copy)
        assert np.all(data_fmodel == rfree.loc[:, rfree.columns != label_name])

    repeat_rfree = rs.utils.add_rfree(
        data_fmodel,
        fraction=fraction,
        ccp4_convention=ccp4_convention,
        inplace=False,
        seed=seed,
    )
    if seed is not None:
        assert np.all(rfree == repeat_rfree)
    else:
        assert not np.all(rfree == repeat_rfree)


@pytest.mark.parametrize("ccp4_convention", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("rfree_key", [None, "custom-rfree-key"])
def test_copy_rfree(data_fmodel, ccp4_convention, inplace, rfree_key):
    """
    Test rs.utils.copy_rfree
    """
    data_copy = data_fmodel.copy()

    # create dataset with rfree flags from which to copy
    data_with_rfree = rs.utils.add_rfree(
        data_fmodel, inplace=False, ccp4_convention=ccp4_convention
    )

    # handle different possible column names for rfree flags
    if rfree_key is not None:
        if ccp4_convention:
            rename_dict = {"FreeR_flag": rfree_key}
        else:
            rename_dict = {"R-free-flags": rfree_key}

        data_with_rfree.rename(columns=rename_dict, inplace=True)
    else:
        if ccp4_convention:
            rfree_key = "FreeR_flag"
        else:
            rfree_key = "R-free-flags"

    data_with_copied_rfree = rs.utils.copy_rfree(
        data_fmodel, data_with_rfree, inplace=inplace, rfree_key=rfree_key
    )

    if inplace:
        assert id(data_with_copied_rfree) == id(data_fmodel)
        assert rfree_key in data_fmodel.columns
        assert np.array_equal(
            data_fmodel[rfree_key].values, data_with_rfree[rfree_key].values
        )
    else:
        assert id(data_with_copied_rfree) != id(data_fmodel)
        assert rfree_key not in data_fmodel.columns
        assert np.array_equal(
            data_with_copied_rfree[rfree_key].values, data_with_rfree[rfree_key].values
        )
        assert np.all(data_fmodel == data_copy)


@pytest.mark.parametrize("rfree_key", [None, "missing key"])
def test_copy_rfree_errors(data_fmodel, rfree_key):
    """
    Test expected ValueErrors for rs.utils.copy_rfree

    When rfree_key=None, copy_rfree searches for columns named
    "R-free-flags" and "FreeR_flag", and throws a ValueError when neither
    is found

    When rfree_key="missing key", copy_rfree throws a ValueError because
    there is no column "missing key"
    """
    with pytest.raises(ValueError):
        rs.utils.copy_rfree(data_fmodel, data_fmodel, rfree_key=rfree_key)
