import unittest
import numpy as np
import reciprocalspaceship as rs

class TestCrystalSeries(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.floatdata = np.zeros(100)
        return
        
    def test_constructor(self):

        # Test empty constructor
        cs = rs.CrystalSeries()
        self.assertIsInstance(cs, rs.CrystalSeries)
        self.assertEqual(cs.name, None)
        self.assertEqual(len(cs), 0)

        # Test constructor with data and name
        cs = rs.CrystalSeries(self.floatdata, name="test")
        self.assertIsInstance(cs, rs.CrystalSeries)
        self.assertTrue(cs.name == "test")
        self.assertEqual(len(cs), len(self.floatdata))

        # Test constructor with data, name, and dtype
        cs = rs.CrystalSeries(self.floatdata, name="test",
                              dtype=rs.PhaseDtype())
        self.assertIsInstance(cs, rs.CrystalSeries)
        self.assertTrue(cs.name == "test")
        self.assertEqual(len(cs), len(self.floatdata))
        self.assertEqual(cs.dtype.mtztype, "P")
        
        return

    def test_constructor_expanddim(self):

        cs = rs.CrystalSeries(self.floatdata, name="test",
                              dtype=rs.PhaseDtype())
        c = cs.to_frame()

        # Test that CrystalSeries expands to Crystal
        self.assertIsInstance(c, rs.Crystal)
        self.assertTrue(len(c.columns), 1)
        self.assertEqual(c.columns[0], "test")

        # If given a name, provided argument becomes column name
        c = cs.to_frame(name="new")
        self.assertIsInstance(c, rs.Crystal)
        self.assertTrue(len(c.columns), 1)
        self.assertEqual(c.columns[0], "new")

        return

if __name__ == '__main__':
    unittest.main()
