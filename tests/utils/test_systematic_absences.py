import pytest
import numpy as np
import reciprocalspaceship as rs
import gemmi

def test_systematic_absences(sgtbx_by_xhm):
    """
    Test rs.utils.hkl_is_absent() using reference data generated from 
    sgtbx
    """
    xhm = sgtbx_by_xhm[0]
    reference = sgtbx_by_xhm[1]
    
    H = reference[['h', 'k', 'l']].to_numpy()
    groupops = gemmi.SpaceGroup(xhm)
    absent = rs.utils.hkl_is_absent(H, groupops)
    reference_absent = reference['is_absent'].to_numpy()
    assert np.array_equal(absent, reference_absent)
