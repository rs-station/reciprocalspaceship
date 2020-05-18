import pytest
from os.path import dirname, abspath, join
import pandas as pd
import gemmi

reference_data = {
    "epsilon": ["..", "data", "epsilon_factors", "epsilon_factors.txt.bz2"],
    "absences": ["..", "data", "systematic_absences", "systematic_absences.txt.bz2"],
    "asu": ["..", "data", "asu", "asu.csv.bz2"],
}

def reference_data_by_xhm(data):
    """
    Generic function for generating a GroupBy iterator over extended 
    Hermann-Mauguin (xhm) symbols
    """
    inFN = abspath(join(dirname(__file__), *reference_data[data]))
    ref = pd.read_csv(inFN)
    return ref.groupby("xhm")

@pytest.fixture(params=[gemmi.SpaceGroup(n) for n in [1, 4, 5, 19, 152]])
def common_spacegroup(request):
    """Yields common space groups for macromolecular crystals"""
    return request.param

@pytest.fixture(params=reference_data_by_xhm("epsilon"))
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
    
@pytest.fixture(params=reference_data_by_xhm("absences"))
def systematic_absences_by_xhm(request):
    """
    Systematic absences grouped by extended Hermann-Manguin symbol for 
    testing.

    Yields
    ------
    Tuple(xhm_str, DataFrame)
        xhm symbol and reference data DataFrame
    """
    return request.param

@pytest.fixture(params=reference_data_by_xhm("asu"))
def asu_by_xhm(request):
    """
    Tests for reflections related to the reciprocal space ASU, 
    grouped by extended Hermann-Manguin symbol for testing.

    Yields
    ------
    Tuple(xhm_str, DataFrame)
        xhm symbol and reference data DataFrame
    """
    return request.param
