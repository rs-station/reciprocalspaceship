import pytest
from os.path import dirname, abspath, join
import numpy as np
import pandas as pd

import reciprocalspaceship as rs
from reciprocalspaceship.algorithms import scale_merged_intensities
from reciprocalspaceship.algorithms.scale_merged_intensities import (
    _acentric_posterior,
    _centric_posterior_quad
)

def load_fw1978():
    """
    Load reference data from French and Wilson, 1978.
    """
    datapath = ["..", "data", "algorithms", "fw1978.csv"]
    inFN = abspath(join(dirname(__file__), *datapath))
    fw1978 = pd.read_csv(inFN, dtype=np.float32)
    return fw1978

def load_hewl():
    """
    Load HEWL diffraction data from APS 24-ID-C
    """
    datapath = ["..", "data", "algorithms", "HEWL_SSAD_24IDC.mtz"]
    inFN = abspath(join(dirname(__file__), *datapath))
    mtz = rs.read_mtz(inFN)
    return mtz
    
@pytest.mark.parametrize("dist", ["Acentric", "Centric"])
@pytest.mark.parametrize(
    "fw1978_refcolumns", [["E(J|I)", "sigma(J|I)"],
                          ["E(F|I)", "sigma(F|I)"]
    ]
)
def test_posteriors_fw1978(dist, fw1978_refcolumns):
    """
    Test posterior distributions used in scale_merged_intensities() 
    against Table 1 from French and Wilson, Acta Cryst. (1978)
    """
    # Compute posterior intensities using distribution for acentric
    # reflections
    fw1978 = load_fw1978()
    I = fw1978["I"]
    SigI = fw1978["SigI"]
    Sigma = 20.*np.ones(len(I), dtype=np.float32)

    if dist == "Acentric":
        mean, stddev = _acentric_posterior(I, SigI, Sigma)
    elif dist =="Centric":
        mean, stddev = _centric_posterior_quad(I, SigI, Sigma)

    fw1978_refcolumns = [" ".join([dist, c]) for c in fw1978_refcolumns]
    ref = fw1978[fw1978_refcolumns]
    
    # Compare intensities
    if "J" in fw1978_refcolumns[0]:
        refI = ref[fw1978_refcolumns[0]].to_numpy()
        refSigI = ref[fw1978_refcolumns[1]].to_numpy()
        assert np.allclose(mean, refI, atol=0.1)
        assert np.allclose(stddev, refSigI, atol=0.01)
        
    # Compare structure factor amplitudes
    else:
        mean = np.sqrt(mean)
        stddev = stddev/(2*mean)
        refF = ref[fw1978_refcolumns[0]].to_numpy()
        refSigF = ref[fw1978_refcolumns[1]].to_numpy()
        assert np.allclose(mean, refF, atol=0.2)
        assert np.allclose(stddev, refSigF, atol=0.1)

@pytest.mark.parametrize("return_intensities", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
def test_scale_merged_intensities_validdata(return_intensities, inplace):
    """
    Confirm scale_merged_intensities() returns all positive values
    """
    mtz = load_hewl().dropna()

    scaled = scale_merged_intensities(mtz, "IMEAN", "SIGIMEAN",
                                      return_intensities=return_intensities,
                                      inplace=inplace)

    # Confirm inplace returns same object if true
    if inplace:
        assert id(scaled) == id(mtz)
    else:
        assert id(scaled) != id(mtz)

    # Confirm intensities are returned if return_intensities is True
    if return_intensities:
        assert isinstance(scaled["FW-IMEAN"].dtype, rs.IntensityDtype)
        assert isinstance(scaled["FW-SIGIMEAN"].dtype, rs.StandardDeviationDtype)
    else:
        assert isinstance(scaled["FW-IMEAN"].dtype, rs.StructureFactorAmplitudeDtype)
        assert isinstance(scaled["FW-SIGIMEAN"].dtype, rs.StandardDeviationDtype)

    assert (scaled["FW-IMEAN"].to_numpy() >= 0).all()
    assert (scaled["FW-SIGIMEAN"].to_numpy() >= 0).all()    
        
