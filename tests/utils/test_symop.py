import gemmi
import numpy as np
import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize("H_even", [True, False])
@pytest.mark.parametrize(
    "op_str", ["x,y,z", "2*x,2*y,2*z", "x,z,y", "1/2*x,y,z", "1/2*x,1/2*y,1/2*z"]
)
def test_apply_to_hkl(hkls, H_even, op_str):
    """
    Test rs.utils.apply_to_hkl() detects symops that yield fractional Miller indices.

    apply_to_hkl() should raise a RuntimeError if the combination of `H` and `op`
    yield fractional Miller indices, and should return new Miller indices all other
    cases.
    """
    if H_even:
        hkls = hkls[~np.any(hkls % 2, axis=1)]

    op = gemmi.Op(op_str)

    if ((np.array(op.rot) / op.DEN) % 1 == 0).all() or H_even:
        H_result = rs.utils.apply_to_hkl(hkls, op)
        H_expected = np.array([op.apply_to_hkl(hkl) for hkl in hkls])
        assert np.array_equal(H_expected, H_result)
        assert H_result.dtype is np.dtype(np.int32)
    else:
        with pytest.raises(RuntimeError):
            H_result = rs.utils.apply_to_hkl(hkls, op)
