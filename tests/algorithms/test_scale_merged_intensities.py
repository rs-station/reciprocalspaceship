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

def test_posteriors_fw1978(data_fw1978_input, data_fw1978_output):
    """
    Test posterior distributions used in scale_merged_intensities() 
    against Table 1 from French and Wilson, Acta Cryst. (1978)
    """
    # Compute posterior intensities using French and Wilson input data
    I     = data_fw1978_input["I"]
    SigI  = data_fw1978_input["SigI"]
    Sigma = data_fw1978_input["Sigma"]

    if "Acentric" in data_fw1978_output.columns[0]:
        mean, stddev = _acentric_posterior(I, SigI, Sigma)
    elif "Centric" in data_fw1978_output.columns[0]:
        mean, stddev = _centric_posterior_quad(I, SigI, Sigma)

    # Compare intensities
    if "J" in data_fw1978_output.columns[0]:
        refI = data_fw1978_output.iloc[:, 0].to_numpy()
        refSigI = data_fw1978_output.iloc[:, 1].to_numpy()
        assert np.allclose(mean, refI, rtol=0.09)
        assert np.allclose(stddev, refSigI, rtol=0.01)
        
    # Compare structure factor amplitudes
    else:
        mean = np.sqrt(mean)
        stddev = stddev/(2*mean)
        refF = data_fw1978_output.iloc[:, 0].to_numpy()
        refSigF =  data_fw1978_output.iloc[:, 1].to_numpy()
        assert np.allclose(mean, refF, rtol=0.3)
        assert np.allclose(stddev, refSigF, rtol=0.2)

def test_centric_posterior(data_fw1978_input):
    """
    Test Gaussian-Legendre quadrature method against scipy quadrature 
    implementation to ensure integration results are consistent with 
    reference implementation. 
    """
    # Compute posterior intensities using French and Wilson input data
    I     = data_fw1978_input["I"]
    SigI  = data_fw1978_input["SigI"]
    Sigma = data_fw1978_input["Sigma"]

    def _centric_posterior_scipy(Iobs, SigIobs, Sigma):
        """
        Reference implementation using scipy quadrature integration to 
        estimate posterior intensities with a wilson prior.
        """
        from scipy.integrate import quad
        
        lower = 0.
        u = Iobs-SigIobs**2/2/Sigma
        upper = np.abs(Iobs) + 10.*SigIobs
        limit = 1000

        Z = np.zeros(len(u))
        for i in range(len(Z)):
            Z[i] = quad(
                lambda J: np.power(J, -0.5)*np.exp(-0.5*((J-u[i])/SigIobs[i])**2),
                lower, upper[i],
            )[0]

        mean = np.zeros(len(u))
        for i in range(len(Z)):
            mean[i] = quad(
                lambda J: J*np.power(J, -0.5)*np.exp(-0.5*((J-u[i])/SigIobs[i])**2),
                lower, upper[i],
            )[0]
        mean = mean/Z

        variance = np.zeros(len(u))
        for i in range(len(Z)):
            variance[i] = quad(
                lambda J: J*J*np.power(J, -0.5)*np.exp(-0.5*((J-u[i])/SigIobs[i])**2),
                lower, upper[i],
            )[0]
        variance = variance/Z - mean**2.

        return mean,np.sqrt(variance)

    mean, stddev = _centric_posterior_quad(I, SigI, Sigma)
    mean_scipy, stddev_scipy = _centric_posterior_scipy(I, SigI, Sigma)
    assert np.allclose(mean, mean_scipy, rtol=0.08)
    assert np.allclose(stddev, stddev_scipy, rtol=0.01)
    
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("output_columns", [None,
                                            ("FW1", "FW2", "FW3", "FW4")])
@pytest.mark.parametrize("mean_intensity_method", ["isotropic", "anisotropic"])
def test_scale_merged_intensities_validdata(data_merged, inplace, output_columns,
                                            mean_intensity_method):
    """
    Confirm scale_merged_intensities() returns all positive values
    """
    scaled = scale_merged_intensities(data_merged, "IMEAN", "SIGIMEAN",
                                      output_columns=output_columns,
                                      inplace=inplace,
                                      mean_intensity_method=mean_intensity_method)

    # Confirm inplace returns same object if true
    if inplace:
        assert id(scaled) == id(data_merged)
    else:
        assert id(scaled) != id(data_merged)

    defaults = ("FW-I", "FW-SIGI", "FW-F", "FW-SIGF")
    if output_columns:
        o1, o2, o3, o4 = output_columns
    else:
        o1, o2, o3, o4 = defaults
        
    # Confirm output columns are of desired types
    assert isinstance(scaled[o1].dtype, rs.IntensityDtype)
    assert isinstance(scaled[o2].dtype, rs.StandardDeviationDtype)
    assert isinstance(scaled[o3].dtype, rs.StructureFactorAmplitudeDtype)
    assert isinstance(scaled[o4].dtype, rs.StandardDeviationDtype)

    # Confirm output columns are strictly positive
    assert (scaled[o1].to_numpy() >= 0).all()
    assert (scaled[o2].to_numpy() >= 0).all()
    assert (scaled[o3].to_numpy() >= 0).all()
    assert (scaled[o4].to_numpy() >= 0).all()

@pytest.mark.parametrize("mean_intensity_method", ["isotropic", "anisotropic"])
def test_scale_merged_intensities_phenix(data_merged, ref_hewl, mean_intensity_method):
    """
    Compare phenix.french_wilson to scale_merged_intensities(). Current
    test criteria are that >95% of F and SigF are within 1%.
    """
    mtz = data_merged.dropna()
    scaled = scale_merged_intensities(mtz, "IMEAN", "SIGIMEAN",
                                      mean_intensity_method=mean_intensity_method)

    # Assert no reflections were dropped
    assert len(scaled) == len(ref_hewl)
    
    # Intensities should be identical
    assert np.array_equal(scaled["IMEAN"].to_numpy(), ref_hewl["I"].to_numpy())
    assert np.array_equal(scaled["SIGIMEAN"].to_numpy(), ref_hewl["SIGI"].to_numpy())
    
    rsF = scaled["FW-F"].to_numpy()
    rsSigF = scaled["FW-SIGF"].to_numpy()
    refF = ref_hewl["F"].to_numpy()
    refSigF = ref_hewl["SIGF"].to_numpy()    
    
    assert (np.isclose(rsF, refF, rtol=0.01).sum()/len(scaled)) >= 0.95
    assert (np.isclose(rsSigF, refSigF, rtol=0.01).sum()/len(scaled)) >= 0.95

@pytest.mark.parametrize("dropna", [True, False])
def test_scale_merged_intensities_dropna(data_hewl_all, dropna):
    """
    Test scale_merged_intensities() using data containing NaNs in different
    columns.
    """
    keys = ["IMEAN", "SIGIMEAN"]
    nan_in_intensity = data_hewl_all[keys].isna().to_numpy().any()

    if nan_in_intensity and not dropna:
        with pytest.raises(ValueError):
            scaled = scale_merged_intensities(data_hewl_all, "IMEAN",
                                              "SIGIMEAN", dropna=dropna)
    else:
        scaled = scale_merged_intensities(data_hewl_all, "IMEAN", "SIGIMEAN",
                                          dropna=dropna)
        assert id(scaled) != id(data_hewl_all)
        assert (scaled["FW-I"].to_numpy() >= 0.).all()
            

    
