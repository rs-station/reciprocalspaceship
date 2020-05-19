import pytest
from os.path import dirname, abspath, join
import pandas as pd
import gemmi

def reference_data_by_xhm():
    """
    Generic function for generating a GroupBy iterator over extended 
    Hermann-Mauguin (xhm) symbols
    """
    reference_data = ["..", "data", "sgtbx", "sgtbx.csv.bz2"]
    inFN = abspath(join(dirname(__file__), *reference_data))
    ref = pd.read_csv(inFN)
    return ref.groupby("xhm")

@pytest.fixture(params=reference_data_by_xhm())
def sgtbx_by_xhm(request):
    """
    sgtbx reference data grouped by extended Hermann-Manguin symbol for 
    testing spacegroup-based methods

    Yields
    ------
    Tuple(xhm_str, DataFrame)
        xhm symbol and reference data DataFrame
    """
    return request.param

@pytest.fixture(params=[gemmi.SpaceGroup(n) for n in [1, 4, 5, 19, 152]])
def common_spacegroup(request):
    """Yields common space groups for macromolecular crystals"""
    return request.param

