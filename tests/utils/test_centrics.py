import gemmi
import numpy as np
import pandas as pd
import pytest

import reciprocalspaceship as rs


def test_is_centric(sgtbx_by_xhm):
    """
    Test rs.utils.is_centric using reference data generated from sgtbx.
    """
    xhm = sgtbx_by_xhm[0]
    reference = sgtbx_by_xhm[1]

    # Drop [0, 0, 0] from test data
    reference = reference.drop(reference.query("h==0 and k==0 and l==0").index)

    H = reference[["h", "k", "l"]].to_numpy()
    ref_centric = reference["is_centric"].to_numpy()
    centric = rs.utils.is_centric(H, xhm)
    assert np.array_equal(centric, ref_centric)
