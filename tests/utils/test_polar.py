from os.path import abspath, dirname, join

import gemmi
import pandas as pd
import pytest

import reciprocalspaceship as rs


def sgtbx_polar_classification():
    """
    Helper function for generating Hermann-Mauguin (xhm) symbols with
    corresponding polar classifications
    """
    data = ["..", "data", "sgtbx", "sgtbx_polar.csv"]
    inFN = abspath(join(dirname(__file__), *data))
    ref = pd.read_csv(inFN)
    return ref.iterrows()


@pytest.fixture(params=sgtbx_polar_classification())
def polar_by_xhm(request):
    """
    sgtbx classifications of whether spacegroup settings are polar.

    Yields
    ------
    Tuple(xhm_str, is_polar_bool)
       xhm symbol and is_polar classification
    """
    i, row = request.param
    return row["xhm"], row["is_polar"]


@pytest.mark.parametrize("use_gemmi_obj", [True, False])
def test_is_polar(polar_by_xhm, use_gemmi_obj):
    """
    Test rs.utils.is_polar() with xhm strings and gemmi.SpaceGroup objects
    """
    xhm, sgtbx_is_polar = polar_by_xhm

    if use_gemmi_obj:
        assert sgtbx_is_polar == rs.utils.is_polar(gemmi.SpaceGroup(xhm))
    else:
        assert sgtbx_is_polar == rs.utils.is_polar(xhm)
