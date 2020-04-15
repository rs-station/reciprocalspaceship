import unittest
import numpy as np
import reciprocalspaceship as rs
import gemmi

class TestASU(unittest.TestCase):

    def test_in_asu(self):

        H = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0],
                      [0, 1, 1],
                      [1, 1, 0],
                      [1, 1, 1]])
        Hneg = H*-1
        sg1  = gemmi.SpaceGroup(1)
        sg19 = gemmi.SpaceGroup(19)

        # Test individual reflections
        bools = rs.utils.asu.in_asu(np.array([[-1, 1, 1]]), sg1)
        self.assertTrue(bools.all())
        self.assertEqual(1, len(bools))
        
        bools = rs.utils.asu.in_asu(np.array([[-1, 1, 1]]), sg19)
        self.assertFalse(bools.all())
        self.assertEqual(1, len(bools))
        
        # Vectorized usage
        bools = rs.utils.asu.in_asu(H, sg1)
        self.assertTrue(bools.all())
        self.assertEqual(len(H), len(bools))
        
        bools = rs.utils.asu.in_asu(H, sg19)
        self.assertTrue(bools.all())
        self.assertEqual(len(H), len(bools))
        
        bools = rs.utils.asu.in_asu(Hneg, sg1)
        self.assertFalse(bools.all())
        self.assertEqual(len(Hneg), len(bools))
        
        bools = rs.utils.asu.in_asu(Hneg, sg19)
        self.assertFalse(bools.all())
        self.assertEqual(len(Hneg), len(bools))
        
        return

    def test_hkl_to_asu(self):

        H = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0],
                      [0, 1, 1],
                      [1, 1, 0],
                      [1, 1, 1]])
        Hneg = H*-1
        sg1  = gemmi.SpaceGroup(1)
        sg19 = gemmi.SpaceGroup(19)

        # return_phase_shifts=False
        H_asu, isymm = rs.utils.asu.hkl_to_asu(H, sg1)
        self.assertTrue(np.array_equal(H, H_asu))
        self.assertTrue(np.array_equal(np.ones(len(H)), isymm))

        H_asu, isymm = rs.utils.asu.hkl_to_asu(Hneg, sg1)
        self.assertTrue(np.array_equal(H, H_asu))
        self.assertTrue(np.array_equal(np.ones(len(H))*2, isymm))

        H_asu, isymm = rs.utils.asu.hkl_to_asu(H, sg19)
        self.assertTrue(np.array_equal(H, H_asu))
        self.assertTrue(np.array_equal(np.ones(len(H)), isymm))

        H_asu, isymm = rs.utils.asu.hkl_to_asu(Hneg, sg19)
        self.assertTrue(np.array_equal(H, H_asu))
        self.assertTrue(np.array_equal(np.array([5, 3, 3, 5, 3, 2]), isymm))

        # return_phase_shifts=True
        H_asu, isymm, phi_coeff, phi_shift = rs.utils.asu.hkl_to_asu(H, sg1, True)
        self.assertTrue(np.array_equal(H, H_asu))
        self.assertTrue(np.array_equal(np.ones(len(H)), isymm))
        self.assertTrue(np.array_equal(np.ones(len(H)), phi_coeff))
        self.assertTrue(np.array_equal(np.zeros(len(H)), phi_shift))

        H_asu, isymm, phi_coeff, phi_shift = rs.utils.asu.hkl_to_asu(Hneg, sg1, True)
        self.assertTrue(np.array_equal(H, H_asu))
        self.assertTrue(np.array_equal(np.ones(len(H))*2, isymm))
        self.assertTrue(np.array_equal(np.ones(len(H))*-1, phi_coeff))
        self.assertTrue(np.array_equal(np.zeros(len(H)), phi_shift))
        
        H_asu, isymm, phi_coeff, phi_shift = rs.utils.asu.hkl_to_asu(H, sg19, True)
        self.assertTrue(np.array_equal(H, H_asu))
        self.assertTrue(np.array_equal(np.ones(len(H)), isymm))
        self.assertTrue(np.array_equal(np.ones(len(H)), phi_coeff))
        self.assertTrue(np.array_equal(np.zeros(len(H)), phi_shift))
        
        H_asu, isymm, phi_coeff, phi_shift = rs.utils.asu.hkl_to_asu(Hneg, sg19, True)
        self.assertTrue(np.array_equal(H, H_asu))
        self.assertTrue(np.array_equal(np.array([5, 3, 3, 5, 3, 2]), isymm))
        phi_coeff_expected = np.array([ 1.,  1.,  1.,  1.,  1., -1.])
        phi_shift_expected = np.array([ -0.,  -0., 180., 180., 180.,  -0.])
        self.assertTrue(np.array_equal(phi_coeff_expected, phi_coeff))
        self.assertTrue(np.array_equal(phi_shift_expected, phi_shift))
        
        return

    
