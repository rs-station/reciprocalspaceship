import pytest
from os.path import dirname, abspath, join
import pandas as pd
import numpy as np
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

@pytest.fixture
def ref_hewl():
    """
    Load phenix.french_wilson output for data_hewl
    """
    datapath = ["..", "data", "algorithms", "phenix_fw_ref.mtz"]
    inFN = abspath(join(dirname(__file__), *datapath))
    mtz = rs.read_mtz(inFN)
    return mtz

def load_fw1978():
    """
    Load reference data from French and Wilson, 1978.
    """
    datapath = ["..", "data", "algorithms", "fw1978.csv"]
    inFN = abspath(join(dirname(__file__), *datapath))
    fw1978 = pd.read_csv(inFN, dtype=np.float32)
    fw1978["Sigma"] = 20.*np.ones(len(fw1978), dtype=np.float32)
    return fw1978

@pytest.fixture
def data_fw1978_input():
    """
    Input data from Table 1 of French and Wilson, 1978
    """
    data = load_fw1978()
    return data[["I",  "SigI", "Sigma"]]

def data_fw1978_intensities(centric):
    """
    Intensity outputs from Table 1 of French and Wilson, 1978
    """
    data = load_fw1978()
    if centric:
        return data[["Centric E(J|I)", "Centric sigma(J|I)"]]
    else:
        return data[["Acentric E(J|I)", "Acentric sigma(J|I)"]]        

def data_fw1978_structurefactors(centric):
    """
    Structure factor outputs from Table 1 of French and Wilson, 1978
    """
    data = load_fw1978()
    if centric:
        return data[["Centric E(F|I)", "Centric sigma(F|I)"]]
    else:
        return data[["Acentric E(F|I)", "Acentric sigma(F|I)"]]        

@pytest.fixture(params=["centric intensities",
                        "acentric intensities",
                        "centric structurefactors",
                        "acentric structurefactors"])
def data_fw1978_output(request):
    """
    Output data from Table 1 of French and Wilson, 1978. Parametrized
    to return both intensities and structure factor amplitudes
    """
    if request.param == "centric intensities":
        return data_fw1978_intensities(centric=True)
    elif request.param == "acentric intensities":
        return data_fw1978_intensities(centric=False)
    elif request.param == "centric structurefactors":
        return data_fw1978_structurefactors(centric=True)
    elif request.param == "acentric structurefactors":
        return data_fw1978_structurefactors(centric=False)

