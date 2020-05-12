import pytest
import numpy as np
import reciprocalspaceship as rs
import pandas as pd
import gemmi

def test_multiplicity_epsilon(epsilon_by_xhm):
    """
    Test rs.utils.compute_structurefactor_multiplicity using reference 
    data generated from sgtbx.
    """
    xhm = epsilon_by_xhm[0]
    reference = epsilon_by_xhm[1]

    # Hexagonal lattices seem to cause issues:
    if ":H" in xhm:
        pytest.xfail("Issues with hexagonal lattices")
    
    H = reference[['h', 'k', 'l']].to_numpy()
    reference_epsilon = reference['epsilon'].to_numpy()
    groupops = gemmi.SpaceGroup(xhm).operations()
    epsilon = rs.utils.compute_structurefactor_multiplicity(H, groupops)
    assert np.array_equal(epsilon, reference_epsilon)

