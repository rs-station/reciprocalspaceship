from os.path import abspath, dirname, join

import gemmi
import pandas as pd
import pytest

import reciprocalspaceship as rs


def sgtbx_polar_classification():
    """
    Helper function for iterating over Hermann-Mauguin (xhm) symbols with
    corresponding polar classifications.

    Yields
    ------
    Tuple(int, pd.Series)
        Row index and pd.Series with values of row in pd.DataFrame
    """
    data = ["..", "data", "sgtbx", "sgtbx_polar.csv"]
    inFN = abspath(join(dirname(__file__), *data))
    ref = pd.read_csv(inFN)
    return ref.iterrows()


@pytest.fixture(params=sgtbx_polar_classification())
def polar_by_xhm(request):
    """
    Fixture function that parametrizes over sgtbx polar classifications
    of each space group setting. This fixture takes the sgtbx_polar_classification()
    iterable and packages each as a (xhm, bool) tuple.

    Yields
    ------
    Tuple(str, bool)
       xhm symbol and whether corresponding space group is polar
    """
    i, row = request.param
    return row["xhm"], row["is_polar"]


@pytest.fixture(params=sgtbx_polar_classification())
def polar_axes_by_xhm(request):
    """
    Fixture function that parametrizes over sgtbx polar classifications
    of each space group setting. This fixture takes the sgtbx_polar_classification()
    iterable and packages each as a (xhm, list) tuple.

    Yields
    ------
    Tuple(str, list)
       xhm symbol and whether the a, b, and c-axes are polar
    """
    i, row = request.param
    polar_axes = [row["is_a_polar"], row["is_b_polar"], row["is_c_polar"]]
    return row["xhm"], polar_axes


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


@pytest.mark.parametrize("use_gemmi_obj", [True, False])
def test_polar_axes(polar_axes_by_xhm, use_gemmi_obj):
    """
    Test rs.utils.polar_axes() with xhm strings and gemmi.SpaceGroup objects
    """
    xhm, expected = polar_axes_by_xhm

    if use_gemmi_obj:
        sg = gemmi.SpaceGroup(xhm)
    else:
        sg = xhm

    # Trigonal spacegroups with rhombohedral Bravais lattices are unsupported
    if xhm.startswith("R ") and xhm.endswith(":R"):
        with pytest.raises(ValueError):
            result = rs.utils.polar_axes(sg)
            return

    else:
        result = rs.utils.polar_axes(sg)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result == expected
