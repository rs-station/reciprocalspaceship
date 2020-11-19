import numpy as np
from scipy.special import ndtr,log_ndtr
from reciprocalspaceship.utils import compute_structurefactor_multiplicity

def _acentric_posterior(Iobs, SigIobs, Sigma):
    """
    Compute the mean and std deviation of the truncated normal 
    French-Wiilson posterior.

    Parameters
    ----------
    Iobs : np.ndarray (float)
        Observed merged refl intensities
    SigIobs : np.ndarray (float)
        Observed merged refl std deviation
    Sigma : np.ndarray (float)
        Average intensity in the resolution bin corresponding to Iobs, SigIobs
    """
    Iobs = np.array(Iobs, dtype=np.float64)
    SigIobs = np.array(SigIobs, dtype=np.float64)
    Sigma = np.array(Sigma, dtype=np.float64)

    def log_phi(x):
        #return np.exp(-0.5*x**2)/np.sqrt(2*np.pi)
        return -0.5*x**2 - np.log(np.sqrt(2*np.pi))

    a = 0.
    s = SigIobs
    u = (Iobs - s**2/Sigma)
    alpha = (a-u)/s
    #Z = 1. - ndtr(alpha)
    log_Z = log_ndtr(-alpha) #<--this is the same thing, I promise
    #mean = u + s * phi(alpha)/Z
    mean = u + np.exp(np.log(s) + log_phi(alpha) - log_Z)
    variance = s**2 * (1 + \
        #alpha*phi(alpha)/Z - \
        alpha * np.exp(log_phi(alpha) - log_Z) - \
        #(phi(alpha)/Z)**2 
        np.exp(2. * log_phi(alpha) - 2. * log_Z)
    )
    return mean, np.sqrt(variance)

def _centric_posterior_quad(Iobs, SigIobs, Sigma, npoints=100):
    """
    Use Gaussian-Legendre quadrature to estimate posterior intensities 
    under a Wilson prior.

    Parameters
    ----------
    Iobs : array (float)
        Observed merged refl intensities
    SigIobs : array (float)
        Observed merged refl std deviation
    Sigma : array (float)
        Average intensity in the resolution bin corresponding to Iobs, SigIobs
    npoints : int
        Number of sample points and weights (must be >= 1)
    """
    Iobs = np.array(Iobs, dtype=np.float64)
    SigIobs = np.array(SigIobs, dtype=np.float64)
    Sigma = np.array(Sigma, dtype=np.float64)

    loc = Iobs-SigIobs**2/2/Sigma
    scale = SigIobs
    upper = np.abs(Iobs) + 10.*SigIobs
    lower = 0.
    upper = upper[:,None]

    grid,weights = np.polynomial.legendre.leggauss(npoints)
    weights = weights[None,:]
    grid = grid[None,:]
    J = (upper - lower)*grid/2. + (upper + lower) / 2.
    prefactor = (upper - lower)/2.
    scale = scale[:,None]
    loc = loc[:,None]

    P = np.power(J, -0.5)*np.exp(-0.5*((J-loc)/scale)**2)
    Z = np.sum(prefactor*weights*P, axis=1)[:,None]
    mean = np.sum(prefactor*weights*J*P/Z, axis=1)
    variance = np.sum(prefactor*weights*J*J*P/Z, axis=1) - mean**2
    return mean,np.sqrt(variance)

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

    if output_columns:
        outputI, outputSigI, outputF, outputSigF = output_columns
    else:
        columns = ["FW-I", "FW-SIGI", "FW-F", "FW-SIGF"]
        outputI, outputSigI, outputF, outputSigF = columns
        
    # Input data for posterior calculations
    I, Sig = ds[intensity_key].to_numpy(), ds[sigma_key].to_numpy()
    if mean_intensity_method == "isotropic":
        dHKL = ds['dHKL'].to_numpy(dtype=np.float64)
        Sigma = mean_intensity_by_resolution(I, dHKL, bins)
    elif mean_intensity_method == "anisotropic":
        Sigma = mean_intensity_by_miller_index(I, ds.get_hkls(), bw)
    multiplicity = compute_structurefactor_multiplicity(ds.get_hkls(), ds.spacegroup)
    Sigma = Sigma * multiplicity

    # Initialize outputs
    ds[outputI] = 0.
    ds[outputSigI] = 0.

    # We will get posterior centric intensities from integration
    mean, std = _centric_posterior_quad(
	ds.loc[ds.CENTRIC, intensity_key].to_numpy(),
	ds.loc[ds.CENTRIC, sigma_key].to_numpy(),
	Sigma[ds.CENTRIC]
    )
    ds.loc[ds.CENTRIC, outputI] = mean
    ds.loc[ds.CENTRIC, outputSigI] = std
    
    # We will get posterior acentric intensities from analytical expressions
    mean, std = _acentric_posterior(
	ds.loc[~ds.CENTRIC, intensity_key].to_numpy(),
	ds.loc[~ds.CENTRIC, sigma_key].to_numpy(),
	Sigma[~ds.CENTRIC]
    )
    ds.loc[~ds.CENTRIC, outputI] = mean
    ds.loc[~ds.CENTRIC, outputSigI] = std

    # Convert dtypes of columns to MTZDtypes
    ds[outputI] = ds[outputI].astype("Intensity")
    ds[outputSigI] = ds[outputSigI].astype("Stddev")
    ds[outputF] = np.sqrt(ds[outputI]).astype("SFAmplitude")
    ds[outputSigF] = (ds[outputSigI]/(2*ds[outputF])).astype("Stddev")

    return ds

if __name__=="__main__": # pragma: no cover
    import reciprocalspaceship as rs
    from sys import argv
    ds = rs.read_mtz(argv[1]).dropna()
    ds = ds.stack_anomalous()
    ds = scale_merged_intensities(ds, "IMEAN", "SIGIMEAN")
    from IPython import embed
    embed(colors='Linux')
