import unittest
from os.path import abspath, dirname, join

import numpy as np

import reciprocalspaceship as rs


class TestRfree(unittest.TestCase):
    def test_add_rfree(self):

        datadir = join(abspath(dirname(__file__)), "../data/fmodel")
        data = rs.read_mtz(join(datadir, "9LYZ.mtz"))

        # Should copy data
        rfree = rs.utils.add_rfree(data)
        self.assertFalse(id(data) == id(rfree))
        self.assertFalse("R-free-flags" in data.columns)
        self.assertTrue("R-free-flags" in rfree.columns)

        # Should keep all other unchanged
        self.assertTrue(np.all(data == rfree.loc[:, rfree.columns != "R-free-flags"]))
        # Should have fewer !=0 as test set
        self.assertTrue(np.sum(rfree.loc[:, "R-free-flags"] != 0)/len(rfree) < 0.15)

        # Test ccp4_convention
        rfree_ccp4 = rs.utils.add_rfree(data, ccp4_convention=True)
        self.assertTrue("FreeR_flag" in rfree_ccp4.columns)

        # Should keep all other unchanged
        self.assertTrue(np.all(data == rfree_ccp4.loc[:, rfree_ccp4.columns != "FreeR_flag"]))
        # Should have fewer ==0 as test set
        self.assertTrue(np.sum(rfree_ccp4.loc[:, "FreeR_flag"] == 0)/len(rfree_ccp4) < 0.15)

        # Test inplace option
        rfree = rs.utils.add_rfree(data, inplace=True)
        self.assertTrue(id(data) == id(rfree))
        self.assertTrue("R-free-flags" in data.columns)
        self.assertTrue("R-free-flags" in rfree.columns)

        return

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
