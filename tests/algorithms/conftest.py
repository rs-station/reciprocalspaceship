import pytest
from os.path import dirname, abspath, join
import reciprocalspaceship as rs

@pytest.fixture
def data_hewl():
    """
    Load HEWL diffraction data from APS 24-ID-C
    """
    datapath = ["..", "data", "algorithms", "HEWL_SSAD_24IDC.mtz"]
    inFN = abspath(join(dirname(__file__), *datapath))
    mtz = rs.read_mtz(inFN)
    return mtz
