import reciprocalspaceship as rs

def compute_doeke_weights(dF, sigdF):
    """
    Compute weights for internal difference map based on Hekstra et al, 
    Nature (2016).

    Parameters
    ----------
    dF : array-like
        Difference structure factor amplitudes
    sigdF : array-like
        Standard deviations of difference structure factor amplitudes
    """
    w = 1. / (1. +
              ((sigdF**2)/(sigdF**2).mean()) +
              0.05*((dF**2)/(dF**2).mean()) )

    return rs.CrystalSeries(w.to_numpy()).astype("Weight")
