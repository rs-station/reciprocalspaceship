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

    # Drop [0, 0, 0] from test data
    reference = reference.drop(reference.query("h==0 and k==0 and l==0").index)

    H = reference[['h', 'k', 'l']].to_numpy()
    ref_centric = reference['is_centric'].to_numpy()
    if sg_type is gemmi.SpaceGroup:
        sg = gemmi.SpaceGroup(xhm)
    elif sg_type is gemmi.GroupOps:
        sg  = gemmi.SpaceGroup(xhm).operations()
    centric = rs.utils.is_centric(H, sg)
    assert np.array_equal(centric, ref_centric)
