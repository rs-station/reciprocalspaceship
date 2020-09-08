import pytest
import numpy as np
import reciprocalspaceship as rs
import pandas as pd
import gemmi

def test_phase_restrictions(sgtbx_by_xhm):
    """
    Test rs.utils.phase_restrictions against reference data generated 
    from sgtbx.
    """
    xhm = sgtbx_by_xhm[0]
    reference = sgtbx_by_xhm[1]

    # Convert reference phase restrictions to canonical output
    H = reference[['h', 'k', 'l']].to_numpy()
    sg = gemmi.SpaceGroup(xhm)
    ref_restrictions =  []
    for h, entry in zip(H, reference["phase_restrictions"].to_list()):
        if entry == "None" or rs.utils.hkl_is_absent([h], sg)[0]:
            ref_restrictions.append([])
        else:
            phases = np.array(entry.split(","), dtype=float)
            phases = rs.utils.canonicalize_phases(np.rad2deg(phases))
            phases.sort()
            ref_restrictions.append(phases)

    restrictions = rs.utils.get_phase_restrictions(H, sg)
    assert len(ref_restrictions) == len(restrictions)
    for ref, test in zip(ref_restrictions, restrictions):
        if ref is []:
            assert ref == test
        else:
            assert np.allclose(np.sin(np.deg2rad(ref)), np.sin(np.deg2rad(np.array(test))))
