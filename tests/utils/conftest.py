import pytest
from os.path import dirname, abspath, join
import pandas as pd
import gemmi

@pytest.fixture(params=[gemmi.SpaceGroup(n) for n in [1, 4, 5, 19, 152]])
def common_spacegroup(request):
    """Yields common space groups for macromolecular crystals"""
    return request.param

def epsilon_factors():
    """
    Load epsilon factors for each extended Hermann-Mauguin (xhm) symbol.

    Returns
    -------
    reference_data : DataFrameGroupBy
        GroupBy iterator for epsilon reference data, grouped by xhm 
        symbol
    """
    datadir = abspath(join(dirname(__file__), "../data/epsilon_factors"))
    inFN = join(datadir, "epsilon_factors.txt.bz2")
    reference_data = pd.read_csv(inFN)
    return reference_data.groupby("xhm")

@pytest.fixture(params=epsilon_factors())
def epsilon_by_xhm(request):
    """
    Epsilon factors grouped by extended Hermann-Manguin symbol for 
    testing.

    Yields
    ------
    Tuple(xhm_str, DataFrame)
        xhm symbol and reference data DataFrame
    """
    return request.param
