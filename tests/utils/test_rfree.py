import unittest
from os.path import abspath, dirname, join

import numpy as np
import pytest

import reciprocalspaceship as rs

@pytest.mark.parametrize("fraction", [0.05, 0.10, 0.15])
@pytest.mark.parametrize("ccp4_convention", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_add_rfree(data_fmodel, fraction, ccp4_convention, inplace):
    '''
    Test rs.utils.add_rfee
    '''
    data_copy = data_fmodel.copy()
    rfree = rs.utils.add_rfree(data_fmodel, fraction=fraction, ccp4_convention=ccp4_convention, inplace=inplace)

    if ccp4_convention:
        label_name = "FreeR_flag"
        assert label_name in rfree.columns
        assert np.sum(rfree.loc[:, label_name] == 0) / len(rfree) < fraction+0.1
    else:
        label_name = "R-free-flags"
        assert label_name in rfree.columns
        assert np.sum(rfree.loc[:, label_name] == 1) / len(rfree) < fraction+0.1

    if inplace:
        assert id(data_fmodel) == id(rfree)
        assert label_name in data_fmodel.columns
        assert np.all(data_copy == rfree.loc[:, rfree.columns != label_name]) 
    else:
        assert id(data_fmodel) != id(rfree)
        assert label_name not in data_fmodel.columns
        assert np.all(data_fmodel == data_copy)
        assert np.all(data_fmodel == rfree.loc[:, rfree.columns != label_name])

class TestRfree(unittest.TestCase):
    def test_copy_rfree(self):

        datadir = join(abspath(dirname(__file__)), "../data/fmodel")
        data = rs.read_mtz(join(datadir, "9LYZ.mtz"))
        data_rfree = rs.utils.add_rfree(data, inplace=False)

        # Test copy of R-free to copy of data
        rfree = rs.utils.copy_rfree(data, data_rfree, inplace=False)
        self.assertFalse(id(data) == id(rfree))
        self.assertFalse("R-free-flags" in data.columns)
        self.assertTrue("R-free-flags" in rfree.columns)
        self.assertTrue(
            np.array_equal(
                rfree["R-free-flags"].values, data_rfree["R-free-flags"].values
            )
        )

        # Test copy of R-free inplace
        rfree = rs.utils.copy_rfree(data, data_rfree, inplace=True)
        self.assertTrue(id(data) == id(rfree))
        self.assertTrue("R-free-flags" in data.columns)
        self.assertTrue("R-free-flags" in rfree.columns)
        self.assertTrue(
            np.array_equal(
                rfree["R-free-flags"].values, data_rfree["R-free-flags"].values
            )
        )

        return
