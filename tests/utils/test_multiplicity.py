import gemmi
import numpy as np
import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize(
    "sg_type",
    [gemmi.SpaceGroup, gemmi.GroupOps],
)
def test_multiplicity_epsilon(sgtbx_by_xhm, sg_type):
    """
    Test rs.utils.compute_structurefactor_multiplicity using reference
    data generated from sgtbx and gemmi.
    """
    xhm = sgtbx_by_xhm[0]
    reference = sgtbx_by_xhm[1]
    # Do not test the 0,0,0 reflection with this test
    # At the moment, gemmi and sgtbx disagree about what the correct value of
    # epsilon_{0,0,0} is for centric spacegroups.
    idx = (reference["h"] != 0) | (reference["k"] != 0) | (reference["l"] != 0)
    assert idx.sum() == len(reference) - 1
    reference = reference[idx]

    H = reference[["h", "k", "l"]].to_numpy()
    sgtbx_epsilon_without_centering = reference[
        "epsilon"
    ].to_numpy()  # This is computed by sgtbx
    gemmi_epsilon = reference["gemmi_epsilon"].to_numpy()  # This is computed by gemmi
    gemmi_epsilon_without_centering = reference[
        "gemmi_epsilon_without_centering"
    ].to_numpy()  # This is computed by gemmi
    if sg_type is gemmi.SpaceGroup:
        sg = gemmi.SpaceGroup(xhm)
    elif sg_type is gemmi.GroupOps:
        sg = gemmi.SpaceGroup(xhm).operations()
    epsilon = rs.utils.compute_structurefactor_multiplicity(H, sg)
    epsilon_without_centering = rs.utils.compute_structurefactor_multiplicity(
        H, sg, include_centering=False
    )
    assert np.array_equal(epsilon_without_centering, sgtbx_epsilon_without_centering)
    assert np.array_equal(epsilon_without_centering, gemmi_epsilon_without_centering)
    assert np.array_equal(epsilon, gemmi_epsilon)
