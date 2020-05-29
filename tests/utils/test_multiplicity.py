import pytest
import numpy as np
import reciprocalspaceship as rs
import pandas as pd
import gemmi

@pytest.mark.parametrize(
    "sg_type", [gemmi.SpaceGroup, gemmi.GroupOps],
)
def test_multiplicity_epsilon(sgtbx_by_xhm, sg_type):
    """
    Test rs.utils.compute_structurefactor_multiplicity using reference 
    data generated from sgtbx.
    """
    xhm = sgtbx_by_xhm[0]
    reference = sgtbx_by_xhm[1]

    H = reference[['h', 'k', 'l']].to_numpy()
    reference_epsilon = reference['epsilon'].to_numpy()
    if sg_type is gemmi.SpaceGroup:
        sg = gemmi.SpaceGroup(xhm)
    elif sg_type is gemmi.GroupOps:
        sg  = gemmi.SpaceGroup(xhm).operations()
    epsilon = rs.utils.compute_structurefactor_multiplicity(H, sg)
    assert np.array_equal(epsilon, reference_epsilon)
