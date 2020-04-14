import unittest
import numpy as np
import reciprocalspaceship as rs

class TestPhases(unittest.TestCase):

    def test_canonicalize_phases_deg(self):

        phases  = np.array([50., -50., 250., -250.])
        results = np.array([50., -50., -110., 110.])
        
        # Test individual values
        p = rs.utils.canonicalize_phases(phases[0], deg=True)
        self.assertEqual(p, results[0])

        p = rs.utils.canonicalize_phases(phases[1], deg=True)
        self.assertEqual(p, results[1])
        
        p = rs.utils.canonicalize_phases(phases[2], deg=True)
        self.assertEqual(p, results[2])

        p = rs.utils.canonicalize_phases(phases[3], deg=True)
        self.assertEqual(p, results[3])

        # Test array of values
        p = rs.utils.canonicalize_phases(phases, deg=True)
        self.assertTrue(np.array_equal(p, results))

        # Test DataSeries
        ds = rs.DataSeries(phases, dtype="Phase")
        p  = rs.utils.canonicalize_phases(ds, deg=True)
        self.assertTrue(np.array_equal(p.values, results))
        
        return

    def test_canonicalize_phases_rad(self):

        phases  = np.deg2rad([50., -50., 250., -250.])
        results = np.deg2rad([50., -50., -110., 110.])
        
        # Test individual values
        p = rs.utils.canonicalize_phases(phases[0], deg=False)
        self.assertTrue(np.isclose(p, results[0]))

        p = rs.utils.canonicalize_phases(phases[1], deg=False)
        self.assertTrue(np.isclose(p, results[1]))
        
        p = rs.utils.canonicalize_phases(phases[2], deg=False)
        self.assertTrue(np.isclose(p, results[2]))

        p = rs.utils.canonicalize_phases(phases[3], deg=False)
        self.assertTrue(np.isclose(p, results[3]))

        # Test array of values
        p = rs.utils.canonicalize_phases(phases, deg=False)
        self.assertTrue(np.isclose(p, results).all())

        # Test DataSeries
        ds = rs.DataSeries(phases, dtype="Phase")
        p  = rs.utils.canonicalize_phases(ds, deg=False)
        self.assertTrue(np.isclose(p, results).all())

        return

    def test_canonicalize_phases_typeerror(self):

        with self.assertRaises(TypeError):
            p = rs.utils.canonicalize_phases(0.0, deg=rs.DataSeries)
        return
