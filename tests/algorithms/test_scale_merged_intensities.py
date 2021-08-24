import numpy as np
import pytest

import reciprocalspaceship as rs
from reciprocalspaceship.algorithms import scale_merged_intensities
from reciprocalspaceship.algorithms.scale_merged_intensities import (
    _french_wilson_posterior_quad,
)


def _acentric_posterior(Iobs, SigIobs, Sigma, npoints=200):
    return _french_wilson_posterior_quad(Iobs, SigIobs, Sigma, False, npoints)


def _centric_posterior(Iobs, SigIobs, Sigma, npoints=200):
    return _french_wilson_posterior_quad(Iobs, SigIobs, Sigma, True, npoints)


def test_posteriors_fw1978(data_fw1978_input, data_fw1978_output):
    """
    Test posterior distributions used in scale_merged_intensities()
    against Table 1 from French and Wilson, Acta Cryst. (1978)
    """
    # Compute posterior intensities using French and Wilson input data
    I = data_fw1978_input["I"]
    SigI = data_fw1978_input["SigI"]
    Sigma = data_fw1978_input["Sigma"]

    if "Acentric" in data_fw1978_output.columns[0]:
        mean, stddev, mean_f, stddev_f = _acentric_posterior(I, SigI, Sigma)
    elif "Centric" in data_fw1978_output.columns[0]:
        mean, stddev, mean_f, stddev_f = _centric_posterior(I, SigI, Sigma)

    # Compare intensities
    if "J" in data_fw1978_output.columns[0]:
        refI = data_fw1978_output.iloc[:, 0].to_numpy()
        refSigI = data_fw1978_output.iloc[:, 1].to_numpy()
        assert np.allclose(mean, refI, rtol=0.05)
        assert np.allclose(stddev, refSigI, rtol=0.05)

    # Compare structure factor amplitudes
    else:
        refF = data_fw1978_output.iloc[:, 0].to_numpy()
        refSigF = data_fw1978_output.iloc[:, 1].to_numpy()
        assert np.allclose(mean_f, refF, rtol=0.05)
        assert np.allclose(stddev_f, refSigF, rtol=0.05)


def test_centric_posterior(data_fw1978_input):
    """
    Test Gaussian-Legendre quadrature method against scipy quadrature
    implementation to ensure integration results are consistent with
    reference implementation.
    """
    # Compute posterior intensities using French and Wilson input data
    I = data_fw1978_input["I"]
    SigI = data_fw1978_input["SigI"]
    Sigma = data_fw1978_input["Sigma"]

    def _centric_posterior_scipy(Iobs, SigIobs, Sigma):
        """
        Reference implementation using scipy quadrature integration to
        estimate posterior intensities with a wilson prior.
        """
        from scipy.integrate import quad

        lower = 0.0
        u = Iobs - SigIobs ** 2 / 2 / Sigma
        upper = np.abs(Iobs) + 10.0 * SigIobs

        Z = np.zeros(len(u))
        for i in range(len(Z)):
            Z[i] = quad(
                lambda J: np.power(J, -0.5)
                * np.exp(-0.5 * ((J - u[i]) / SigIobs[i]) ** 2),
                lower,
                upper[i],
            )[0]

        mean = np.zeros(len(u))
        for i in range(len(Z)):
            mean[i] = quad(
                lambda J: J
                * np.power(J, -0.5)
                * np.exp(-0.5 * ((J - u[i]) / SigIobs[i]) ** 2),
                lower,
                upper[i],
            )[0]
        mean = mean / Z

        variance = np.zeros(len(u))
        for i in range(len(Z)):
            variance[i] = quad(
                lambda J: J
                * J
                * np.power(J, -0.5)
                * np.exp(-0.5 * ((J - u[i]) / SigIobs[i]) ** 2),
                lower,
                upper[i],
            )[0]
        variance = variance / Z - mean ** 2.0

        return mean, np.sqrt(variance)

    mean, stddev, mean_f, stddev_f = _centric_posterior(I, SigI, Sigma)
    mean_scipy, stddev_scipy = _centric_posterior_scipy(I, SigI, Sigma)
    assert np.allclose(mean, mean_scipy, rtol=0.1)
    assert np.allclose(stddev, stddev_scipy, rtol=0.05)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("output_columns", [None, ("FW1", "FW2", "FW3", "FW4")])
@pytest.mark.parametrize("mean_intensity_method", ["isotropic", "anisotropic"])
def test_scale_merged_intensities_validdata(
    hewl_merged, inplace, output_columns, mean_intensity_method
):
    """
    Confirm scale_merged_intensities() returns all positive values
    """
    scaled = scale_merged_intensities(
        hewl_merged,
        "IMEAN",
        "SIGIMEAN",
        output_columns=output_columns,
        inplace=inplace,
        mean_intensity_method=mean_intensity_method,
    )

    # Confirm inplace returns same object if true
    if inplace:
        assert id(scaled) == id(hewl_merged)
    else:
        assert id(scaled) != id(hewl_merged)

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
def test_scale_merged_intensities_phenix(hewl_merged, ref_hewl, mean_intensity_method):
    """
    Compare phenix.french_wilson to scale_merged_intensities(). Current
    test criteria are that >95% of I, SigI, F, and SigF are within 2%.
    """
    mtz = hewl_merged.dropna()
    scaled = scale_merged_intensities(
        mtz, "IMEAN", "SIGIMEAN", mean_intensity_method=mean_intensity_method
    )

    # Assert no reflections were dropped
    assert len(scaled) == len(ref_hewl)

    # Intensities should be identical
    assert np.array_equal(scaled["IMEAN"].to_numpy(), ref_hewl["I"].to_numpy())
    assert np.array_equal(scaled["SIGIMEAN"].to_numpy(), ref_hewl["SIGI"].to_numpy())

    rsF = scaled["FW-F"].to_numpy()
    rsSigF = scaled["FW-SIGF"].to_numpy()
    refF = ref_hewl["F"].to_numpy()
    refSigF = ref_hewl["SIGF"].to_numpy()

    rsI = scaled["FW-I"].to_numpy()
    rsSigI = scaled["FW-SIGI"].to_numpy()
    refI = ref_hewl["I"].to_numpy()
    refSigI = ref_hewl["SIGI"].to_numpy()

    assert (np.isclose(rsI, refI, rtol=0.02).sum() / len(scaled)) >= 0.95
    assert (np.isclose(rsSigI, refSigI, rtol=0.02).sum() / len(scaled)) >= 0.95

    assert (np.isclose(rsF, refF, rtol=0.02).sum() / len(scaled)) >= 0.95
    assert (np.isclose(rsSigF, refSigF, rtol=0.02).sum() / len(scaled)) >= 0.95


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
            scaled = scale_merged_intensities(
                data_hewl_all, "IMEAN", "SIGIMEAN", dropna=dropna
            )
    else:
        scaled = scale_merged_intensities(
            data_hewl_all, "IMEAN", "SIGIMEAN", dropna=dropna
        )
        assert id(scaled) != id(data_hewl_all)
        assert (scaled["FW-I"].to_numpy() >= 0.0).all()


@pytest.mark.parametrize("ref_method", ["mcmc", "cctbx", "mcmc vs cctbx"])
def test_fw_posterior_quad(ref_method, data_fw_cctbx, data_fw_mcmc):
    """
    Test the _french_wilson_posterior_quad function directly against cctbx output.
    The cctbx implementations can be found
     - `acentric <https://github.com/cctbx/cctbx_project/blob/8db5aedd7d0897bcea82b9c8dc2d21976437435f/cctbx/french_wilson.py#L92>`_
     - `centric <https://github.com/cctbx/cctbx_project/blob/8db5aedd7d0897bcea82b9c8dc2d21976437435f/cctbx/french_wilson.py#L128>`_
    """
    if ref_method == "cctbx":
        I, SigI, Sigma, J, SigJ, F, SigF, Centric = data_fw_cctbx.to_numpy(np.float64).T
        Centric = Centric.astype(bool)
        rtol = 0.06
        rs_J, rs_SigJ, rs_F, rs_SigF = _french_wilson_posterior_quad(
            I, SigI, Sigma, Centric
        )
    elif ref_method == "mcmc":
        I, SigI, Sigma, J, SigJ, F, SigF, Centric = data_fw_mcmc.to_numpy(np.float64).T
        Centric = Centric.astype(bool)
        rtol = 0.025
        rs_J, rs_SigJ, rs_F, rs_SigF = _french_wilson_posterior_quad(
            I, SigI, Sigma, Centric
        )
    elif ref_method == "mcmc vs cctbx":
        I, SigI, Sigma, J, SigJ, F, SigF, Centric = data_fw_mcmc.to_numpy(np.float64).T
        Centric = Centric.astype(bool)
        _, _, _, rs_J, rs_SigJ, rs_F, rs_SigF, _ = data_fw_cctbx.to_numpy(np.float64).T
        rtol = 0.06

    assert np.allclose(rs_J, J, rtol=rtol)
    assert np.allclose(rs_SigJ, SigJ, rtol=rtol)
    assert np.allclose(rs_F, F, rtol=rtol)
    assert np.allclose(rs_SigF, SigF, rtol=rtol)
