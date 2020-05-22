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
