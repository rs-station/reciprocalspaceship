import pytest
import numpy as np
import reciprocalspaceship as rs
import gemmi

@pytest.mark.parametrize(
    "refls", [
        np.array([[1, 1, 1]]),
        np.array([[0, 0, 1],
                  [1, 0, 0],
                  [1, 1, 0],
                  [1, 1, 1]])
    ]
)
def test_in_asu(refls, common_spacegroup):
    # Test whether Miller indices are in reciprocal space ASU
    
    bools = rs.utils.asu.in_asu(refls, common_spacegroup)
    assert bools.all()
    assert len(bools) == len(refls)

    bools = rs.utils.asu.in_asu(-1*refls, common_spacegroup)
    assert not bools.all()
    assert len(bools) == len(refls)

@pytest.mark.parametrize(
    "refls", [
        np.array([[1, 1, 1]]),
        np.array([[0, 0, 1],
                  [1, 0, 0],
                  [1, 1, 0],
                  [1, 1, 1]])
    ]
)
@pytest.mark.parametrize("return_phase_shifts", [True, False])
def test_hkl_to_asu(refls, common_spacegroup, return_phase_shifts):

    # In ASU case
    if return_phase_shifts:
       Hasu, isymm, phic, phishift = rs.utils.asu.hkl_to_asu(refls,
                                                             common_spacegroup,
                                                             return_phase_shifts)
       assert np.array_equal(np.ones(len(refls)), phic)
       assert np.array_equal(np.zeros(len(refls)), phishift)
    else:
       Hasu, isymm = rs.utils.asu.hkl_to_asu(refls, common_spacegroup,
                                             return_phase_shifts)

    assert np.array_equal(refls, Hasu)
    assert np.array_equal(np.ones(len(refls)), isymm)

    # Out of ASU case
    if return_phase_shifts:
       Hasu, isymm, phic, phishift = rs.utils.asu.hkl_to_asu(-1*refls,
                                                             common_spacegroup,
                                                             return_phase_shifts)
       ops_plus  = [ op for op in common_spacegroup.operations() ]
       ops_minus = [ op.negated() for op in common_spacegroup.operations() ]
       ops = np.array(list(zip(ops_plus, ops_minus))).flatten()
       phishift_expected = [ ops[i-1].phase_shift(h) for h, i in zip(Hasu, isymm) ]
       phishift_expected = rs.utils.canonicalize_phases(np.rad2deg(phishift_expected))
       assert np.isclose(phishift_expected, rs.utils.canonicalize_phases(phishift)).all()
       
    else:
       Hasu, isymm = rs.utils.asu.hkl_to_asu(-1*refls, common_spacegroup,
                                             return_phase_shifts)

    assert np.array_equal(refls, Hasu)
