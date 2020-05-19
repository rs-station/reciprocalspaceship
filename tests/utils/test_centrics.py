import pytest
import numpy as np
import reciprocalspaceship as rs
import pandas as pd
import gemmi

@pytest.mark.parametrize(
    "sg_type", [gemmi.SpaceGroup, gemmi.GroupOps],
)
def test_is_centric(sgtbx_by_xhm, sg_type):
    """
    Test rs.utils.is_centric using reference data generated from sgtbx.
    """
    xhm = sgtbx_by_xhm[0]
    reference = sgtbx_by_xhm[1]

    H = reference[['h', 'k', 'l']].to_numpy()
    ref_centric = reference['is_centric'].to_numpy()
    if sg_type is gemmi.SpaceGroup:
        sg = gemmi.SpaceGroup(xhm)
    elif sg_type is gemmi.GroupOps:
        sg  = gemmi.SpaceGroup(xhm).operations()
    centric = rs.utils.is_centric(H, sg)
    assert np.array_equal(centric, ref_centric)
