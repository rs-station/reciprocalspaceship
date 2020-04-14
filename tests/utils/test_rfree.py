import unittest
import numpy as np
from os.path import dirname, abspath, join
import reciprocalspaceship as rs


class TestRfree(unittest.TestCase):

    def test_add_rfree(self):

        datadir = join(abspath(dirname(__file__)), '../data/fmodel')
        data = rs.read_mtz(join(datadir, '9LYZ.mtz'))

        # Should copy data
        rfree = rs.utils.add_rfree(data)
        self.assertFalse(id(data) == id(rfree))
        self.assertFalse("R-free-flags" in data.columns)
        self.assertTrue("R-free-flags" in rfree.columns)

        # Test inplace option
        rfree = rs.utils.add_rfree(data, inplace=True)
        self.assertTrue(id(data) == id(rfree))
        self.assertTrue("R-free-flags" in data.columns)
        self.assertTrue("R-free-flags" in rfree.columns)

        return

    def test_copy_rfree(self):

        datadir = join(abspath(dirname(__file__)), '../data/fmodel')
        data = rs.read_mtz(join(datadir, '9LYZ.mtz'))
        data_rfree = rs.utils.add_rfree(data, inplace=False)

        # Test copy of R-free to copy of data
        rfree = rs.utils.copy_rfree(data, data_rfree, inplace=False)
        self.assertFalse(id(data) == id(rfree))
        self.assertFalse("R-free-flags" in data.columns)
        self.assertTrue("R-free-flags" in rfree.columns)
        self.assertTrue(np.array_equal(rfree["R-free-flags"].values,
                                       data_rfree["R-free-flags"].values))
        
        # Test copy of R-free inplace
        rfree = rs.utils.copy_rfree(data, data_rfree, inplace=True)
        self.assertTrue(id(data) == id(rfree))
        self.assertTrue("R-free-flags" in data.columns)
        self.assertTrue("R-free-flags" in rfree.columns)
        self.assertTrue(np.array_equal(rfree["R-free-flags"].values,
                                       data_rfree["R-free-flags"].values))

        return
