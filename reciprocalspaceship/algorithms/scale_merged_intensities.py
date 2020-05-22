import numpy as np
from scipy.special import erf
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
        Average intensity in the resolution bin corresponding to Iobs, 
        SigIobs
    """
    def Phi(x):
        return 0.5*(1 + erf(x/np.sqrt(2.)))

    def phi(x):
        return np.exp(-0.5*x**2)/np.sqrt(2*np.pi)

    eps = 0.

    a = 0.
    b = 1e300
    s = SigIobs
    u = (Iobs - s**2/Sigma)
    alpha = (a-u)/(s + eps)
    beta = (b-u)/(s + eps)
    Z = Phi(beta) - Phi(alpha) + eps
    mean = u + s * (phi(alpha) - phi(beta))/Z
    variance = s**2 * (1 + (alpha*phi(alpha) - beta*phi(beta))/Z - ((phi(alpha) - phi(beta))/Z)**2 )
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

def scale_merged_intensities(ds, intensity_key, sigma_key, output_prefix="FW-",
                             bins=50, return_intensities=False, inplace=False):
    """
    Scales merged intensities using Bayesian statistics in order to 
    estimate structure factor amplitudes. This method is based on French
    and Wilson, Acta Cryst. (1978), and is useful for improving the 
    estimates of negative and small intensities in order to ensure that 
    structure factor moduli are strictly positive.

    The mean and standard deviation of acentric reflections are computed
    analytically from a truncated normal distribution. The mean and 
    standard deviation for centric reflections are computed by numerical
    integration of the posterior intensity distribution under a Wilson 
    prior, and then by interpolation with a kernel smoother.

    Notes
    -----
    This method follows the same approach as French and Wilson, with 
    the following modifications:
    - Numerical integration under a  Wilson prior is used to estimate the
      mean and standard deviation of centric reflections at runtime, rather
      than using cached results and a look-up table
    - Same procedure is used for all centric reflections; original work 
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
    output_prefix : str
        Prefix to be prepended to intensity_key and sigma_key for output 
        column labels
    bins : int or array
        Either an integer number of n bins. Or an array of bin edges with shape==(n, 2)
    return_intensities : bool
        If True, intensities are returned. If False, structure factor 
        amplitudes are returned. 
    inplace : bool
        Whether to modify the DataSet in place or create a copy

    Returns
    -------
    DataSet
        
    """
    if not inplace:
        ds = ds.copy()
    if 'dHKL' not in ds:
        ds.compute_dHKL(inplace=True)
    if 'CENTRIC' not in ds:
        ds.label_centrics(inplace=True)

    d = ds.compute_dHKL().dHKL.to_numpy()**-2.
    if isinstance(bins, int):
        binedges = np.percentile(d, np.linspace(0, 100, bins+1))
        binedges = np.vstack((binedges[:-1], binedges[1:]))

    # If intensity_key or sigma_key are not columns in ds, KeyError is
    # raised
    I, Sig = ds[intensity_key].to_numpy(), ds[sigma_key].to_numpy()

    idx = (d[:,None] > binedges[0]) & (d[:,None] < binedges[1])
    SigmaMean = (I[:,None]*idx).sum(0)/idx.sum(0)
    dmean = (d[:,None]*idx).sum(0)/idx.sum(0)

    # Use kernel smoother instead of linear interpolation.
    h = (d.max() - d.min())/len(dmean) # Bandwidth is roughly the spacing of estimates
    W = np.exp(-0.5*((d[:,None] - dmean)/h)**2)
    W = W/W.sum(1)[:,None]
    Sigma = W@SigmaMean

    multiplicity = compute_structurefactor_multiplicity(ds.get_hkls(), ds.spacegroup)
    Sigma = Sigma * multiplicity

    outval_label = output_prefix + intensity_key
    outerr_label = output_prefix + sigma_key
    
    ds[outval_label] = 0.
    ds[outerr_label] = 0.

    # We will get posterior centric intensities from integration
    mean, std = _centric_posterior_quad(
	ds.loc[ds.CENTRIC, intensity_key].to_numpy(),
	ds.loc[ds.CENTRIC, sigma_key].to_numpy(),
	Sigma[ds.CENTRIC]
    )

    ds.loc[ds.CENTRIC, outval_label] = mean
    ds.loc[ds.CENTRIC, outerr_label] = std
    
    # We will get posterior acentric intensities from analytical expressions
    mean, std = _acentric_posterior(
	ds.loc[~ds.CENTRIC, intensity_key].to_numpy(),
	ds.loc[~ds.CENTRIC, sigma_key].to_numpy(),
	Sigma[~ds.CENTRIC]
    )
    ds.loc[~ds.CENTRIC, outval_label] = mean
    ds.loc[~ds.CENTRIC, outerr_label] = std

    if return_intensities:
        ds[outval_label] = ds[outval_label].astype("Intensity")
        ds[outerr_label] = ds[outerr_label].astype("Stddev")
    else:
        ds[outerr_label] = (ds[outerr_label]/(2*ds[outval_label])).astype("Stddev")
        ds[outval_label] = np.sqrt(ds[outval_label]).astype("SFAmplitude")

    return ds
