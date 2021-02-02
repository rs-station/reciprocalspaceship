import numpy as np
import reciprocalspaceship as rs
from scipy.special import logsumexp
from scipy.stats import gamma,norm


def _french_wilson_posterior_quad(Iobs, SigIobs, Sigma, centric, npoints=100):
    """
    Compute the mean and std deviation of the French-Wilson posterior using 
    Chebyshev-Gauss quadrature. 

    Parameters
    ----------
    Iobs : np.ndarray (float)
        Observed merged refl intensities
    SigIobs : np.ndarray (float)
        Observed merged refl std deviation
    Sigma : np.ndarray (float)
        Average intensity in the resolution bin corresponding to Iobs, SigIobs
    centric : np.ndarray (bool)
        Array where `True` corresponds to centric reflections and `False` acentric. 
    npoints : int
        Number of grid points at which to evaluate the integrand. See the `deg` 
        parameter in the numpy 
        `documentation <https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html`_
        for more details.

    Returns
    -------
    mean_I : np.ndarray (float)
        Mean posterior intensity
    std_I : np.ndarray (float)
        Standard deviation of posterior intensity
    mean_F : np.ndarray (float)
        Mean posterior structure factor amplitude
    std_F :np.ndarray (float)
        Standard deviation of posterior structure factor amplitude
    """
    Iobs = np.array(Iobs, dtype=np.float64)
    SigIobs = np.array(SigIobs, dtype=np.float64)
    Sigma = np.array(Sigma, dtype=np.float64)

    #Integration window based on the normal, likelihood distribution
    window_size = 20. #In standard devs
    Jmin = Iobs - window_size*SigIobs/2.
    Jmin = np.maximum(0., Jmin)
    Jmax = Jmin + window_size*SigIobs

    #Prior distribution paramters
    a = np.where(centric, 0.5, 1.)
    scale = np.where(centric, 2.*Sigma, Sigma)

    #Quadrature grid points
    grid,weights = np.polynomial.chebyshev.chebgauss(npoints)

    #Change of variables for generalizing chebgauss
    # Equivalent to: logweights = log(sqrt(1-grid**2.)*w)
    logweights = (0.5*(np.log(1-grid) + np.log(1+grid)) + np.log(weights))[None,:] 
    J = (Jmax - Jmin)[:,None] * grid / 2. + (Jmax + Jmin)[:,None] / 2.
    logJ = np.nan_to_num(np.log(J), -np.inf)
    log_prefactor = np.log(Jmax - Jmin) - np.log(2.0)

    #Log prior (Wilson's prior for intensities)
    logP = gamma.logpdf(J, np.expand_dims(a, axis=-1), scale=scale[:,None])
    
    #Log likelihood (Normal about Iobs)
    logL = norm.logpdf(J, loc=Iobs[:,None], scale=SigIobs[:,None])

    #Compute partition function
    logZ = log_prefactor + logsumexp(logweights + logP + logL, axis=1) 

    #Compute expected value and variance of intensity
    log_mean = log_prefactor + logsumexp(logweights + logJ + logP + logL - logZ[:,None], axis=1)
    mean = np.exp(log_mean)
    variance = np.exp(log_prefactor + logsumexp(logweights+2.*logJ+logP + logL - logZ[:,None], axis=1)) - mean**2

    #Compute expected value and variance of structure factor amplitude
    logF = 0.5*logJ #Change variables
    log_mean_F = log_prefactor + logsumexp(logweights + logF + logP + logL - logZ[:,None], axis=1)
    mean_F = np.exp(log_mean_F)
    variance_F = mean - mean_F**2
    return mean, np.sqrt(variance), mean_F, np.sqrt(variance_F)

def mean_intensity_by_miller_index(I, H, bandwidth):
    """
    Use a gaussian kernel smoother to compute mean intensities as a function of miller index.

    Parameters
    ----------
    I : array
        Array of observed intensities
    H : array
        Nx3 array of miller indices
    bandwidth : float(optional)
        Kernel bandwidth in miller units

    Returns
    -------
    Sigma : array
        Array of point estimates for the mean intensity at each miller index in H.
    """
    H = np.array(H, dtype=np.float32)
    I = np.array(I, dtype=np.float32)
    bandwidth = np.float32(bandwidth)**2.
    n = len(I)

    S = np.zeros(n)
    for i in range(n):
        K = np.exp(-0.5*((H - H[i])*(H - H[i])).sum(1)/bandwidth)
        S[i] = (I*K).sum()/K.sum()

    return S

def mean_intensity_by_resolution(I, dHKL, bins=50, gridpoints=None):
    """
    Use a gaussian kernel smoother to compute mean intensities as a function of resolution.
    The kernel smoother is evaulated over the specified number of gridpoints and then interpolated. 
    Kernel bandwidth is derived from `bins` as follows
    >>> X = dHKL**-2
    bw = (X.max() - X.min)/bins

    Parameters
    ----------
    I : array
        Array of observed intensities
    dHKL : array
        Array of reflection resolutions
    bins : float(optional)
        "bins" is used to determine the kernel bandwidth.
    gridpoints : int(optional)
        Number of gridpoints at which to estimate the mean intensity. This will default to 20*bins

    Returns
    -------
    Sigma : array
        Array of point estimates for the mean intensity at resolution in dHKL.
    """
    #Use double precision
    I = np.array(I, dtype=np.float64)
    dHKL = np.array(dHKL, dtype=np.float64)

    if gridpoints is None:
        gridpoints = int(bins*20)

    X = dHKL**-2.
    bw = (X.max() - X.min())/bins

    #Evaulate the kernel smoother over grid points
    grid = np.linspace(X.min(), X.max(), gridpoints)
    K = np.exp(-0.5*((X[:,None] - grid[None,:])/bw)**2.)
    K = K/K.sum(0)
    protos = I@K

    #Use a kernel smoother to interpolate the grid points
    bw = grid[1] - grid[0]
    K = np.exp(-0.5*((X[:,None] - grid[None,:])/bw)**2.)
    K = K/K.sum(1)[:,None]
    Sigma = K@protos

    return Sigma

def scale_merged_intensities(ds, intensity_key, sigma_key, output_columns=None,
                             dropna=True, inplace=False, mean_intensity_method="isotropic",
                             bins=100, bw=2.0):
    """
    Scales merged intensities using Bayesian statistics in order to 
    estimate structure factor amplitudes. This method is based on the approach
    by French and Wilson [1]_, and is useful for improving the estimates 
    of negative and small intensities in order to ensure that structure 
    factor moduli are strictly positive.

    The mean and standard deviation of acentric reflections are computed
    analytically from a truncated normal distribution. The mean and 
    standard deviation for centric reflections are computed by numerical
    integration of the posterior intensity distribution under a Wilson 
    prior, and then by interpolation with a kernel smoother.

    Notes
    -----
        This method follows the same approach as French and Wilson, with 
        the following modifications:

        * Numerical integration under a Wilson prior is used to estimate the
          mean and standard deviation of centric reflections at runtime, 
          rather than using precomputed results and a look-up table.
        * Same procedure is used for all centric reflections; original work 
          handled high intensity centric reflections differently.

    Parameters
    ----------
    ds : DataSet
        Input DataSet containing columns with intensity_key and sigma_key
        labels
    intensity_key : str
        Column label for intensities to be scaled
    sigma_key : str
        Column label for error estimates of intensities being scaled
    output_columns : list or tuple of column names
        Column labels to be added to ds for recording scaled I, SigI, F, 
        and SigF, respectively. output_columns must have len=4.
    dropna : bool
        Whether to drop reflections with NaNs in intensity_key or sigma_key
        columns
    inplace : bool
        Whether to modify the DataSet in place or create a copy
    mean_intensity_method : str ["isotropic" or "anisotropic"]
        If "isotropic", mean intensity is determined by resolution bin.
        If "anisotropic", mean intensity is determined by Miller index 
        using provided bandwidth.
    bins : int or array
        Either an integer number of n bins. Or an array of bin edges with 
        shape==(n, 2). Only affects output if mean_intensity_method is
        \"isotropic\".
    bw : float
        Bandwidth to use for computing anisotropic mean intensity. This 
        parameter controls the distance that each reflection impacts in 
        reciprocal space. Only affects output if mean_intensity_method is
        \"anisotropic\".

    Returns
    -------
    DataSet
        DataSet with 4 additional columns corresponding to scaled I, SigI,
        F, and SigF. 

    References
    ----------
    .. [1] French S. and Wilson K. \"On the Treatment of Negative Intensity
       Observations,\" Acta Cryst. A34 (1978).
    """
    if not inplace:
        ds = ds.copy()

    # Sanitize input or check for invalid reflections
    if dropna:
        ds.dropna(subset=[intensity_key, sigma_key], inplace=True)
    else:
        if ds[[intensity_key, sigma_key]].isna().to_numpy().any():
            raise ValueError(f"Input {ds.__class__.__name__} contains NaNs "
                             f"in columns '{intensity_key}' and/or 'sigma_key'. "
                             f"Please fix these input values, or run with dropna=True")
        
    # Accessory columns needed for algorithm
    if 'dHKL' not in ds:
        ds.compute_dHKL(inplace=True)
    if 'CENTRIC' not in ds:
        ds.label_centrics(inplace=True)

    if output_columns is None:
        output_columns = ["FW-I", "FW-SIGI", "FW-F", "FW-SIGF"]
    outputI, outputSigI, outputF, outputSigF = output_columns

    multiplicity = ds.compute_multiplicity().EPSILON.to_numpy()
    # Input data for posterior calculations
    I, Sig = ds[intensity_key].to_numpy(), ds[sigma_key].to_numpy()
    if mean_intensity_method == "isotropic":
        dHKL = ds['dHKL'].to_numpy(dtype=np.float64)
        Sigma = mean_intensity_by_resolution(I/multiplicity, dHKL, bins)*multiplicity
    elif mean_intensity_method == "anisotropic":
        Sigma = mean_intensity_by_miller_index(I/multiplicity, ds.get_hkls(), bw)*multiplicity

    # Initialize outputs
    ds[outputI] = 0.
    ds[outputSigI] = 0.
    
    mean_I, std_I, mean_F, std_F = _french_wilson_posterior_quad(
	ds[intensity_key].to_numpy(),
	ds[sigma_key].to_numpy(),
	Sigma,
        ds.CENTRIC.to_numpy()
    )

    # Convert dtypes of columns to MTZDtypes
    ds[outputI] = rs.DataSeries(mean_I, index=ds.index, dtype="Intensity")
    ds[outputSigI] = rs.DataSeries(std_I, index=ds.index, dtype="Stddev")
    ds[outputF] = rs.DataSeries(mean_F, index=ds.index, dtype="SFAmplitude")
    ds[outputSigF] = rs.DataSeries(std_F, index=ds.index, dtype="Stddev")

    return ds

if __name__=="__main__": # pragma: no cover
    import reciprocalspaceship as rs
    from sys import argv
    ds = rs.read_mtz(argv[1]).dropna()
    ds = ds.stack_anomalous()
    ds = scale_merged_intensities(ds, "IMEAN", "SIGIMEAN")
    from IPython import embed
    embed(colors='Linux')
