import unittest
import numpy as np
from os.path import dirname, abspath, join
import reciprocalspaceship as rs

class TestStructureFactors(unittest.TestCase):

    def test_to_structurefactor(self):

        # Given numpy arrays for SFAmps and Phases, should return
        # complex numbers
        sfs1 = rs.utils.to_structurefactor(np.random.rand(10),
                                           np.random.rand(10))
        self.assertEqual(len(sfs1), 10)
        self.assertTrue(np.iscomplexobj(sfs1))

        # Lists should work too
        sfs2 = rs.utils.to_structurefactor([1], [10])
        self.assertEqual(len(sfs2), 1)
        self.assertTrue(np.iscomplexobj(sfs2))

        # Empty lists as arguments should return empty array
        sfs3 = rs.utils.to_structurefactor([], [])
        self.assertEqual(len(sfs3), 0)
        self.assertTrue(np.iscomplexobj(sfs3))

        # Given single SFAmp + Phase, should return single complex value
        sf = rs.utils.to_structurefactor(1.4, 130)
        self.assertEqual(sf.size, 1)
        self.assertTrue(np.iscomplexobj(sf))

        # Test DataSeries objects as arguments
        datadir = join(abspath(dirname(__file__)), 'data/fmodel')
        data = rs.read_mtz(join(datadir, '9LYZ.mtz'))
        sfs = rs.utils.to_structurefactor(data["FMODEL"],
                                          data["PHIFMODEL"])
        self.assertEqual(len(sfs), len(data))
        self.assertTrue(np.iscomplexobj(sfs))

        return

    def test_from_structurefactor(self):

        # Given complex numpy array, should return SFAmps and phases
        sfs = np.random.rand(10)*np.exp(1j*np.random.rand(10))
        sf, phase = rs.utils.from_structurefactor(sfs)
        self.assertEqual(len(sf), len(sfs))
        self.assertEqual(len(phase), len(sfs))
        self.assertIsInstance(sf, rs.DataSeries)
        self.assertIsInstance(phase, rs.DataSeries)
        self.assertEqual(sf.dtype.mtztype, "F")
        self.assertEqual(phase.dtype.mtztype, "P")

        return

    
if __name__ == '__main__':
    unittest.main()
