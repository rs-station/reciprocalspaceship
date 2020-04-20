from .formats import precognition, mtz

def read_mtz(mtzfile):
    """
    Populate the crystal object with data from an MTZ reflection file.
    
    Parameters
    ----------
    mtzfile : str or file
        name of an mtz file or a file object
    """
    return mtz.read(mtzfile)

def read_precognition(hklfile, a=None, b=None, c=None, alpha=None,
                      beta=None, gamma=None, sg=None, logfile=None):
    """
    Initialize attributes and populate the crystal object with data 
    from a HKL file of reflections. This is the output format used 
    by Precognition, a Laue analysis suite.

    Parameters
    ----------
    hklfile : str or file
        name of an hkl file or a file object
    a : float
        edge length, a, of the unit cell
    b : float
        edge length, b, of the unit cell
    c : float
        edge length, c, of the unit cell
    alpha : float
        interaxial angle, alpha, of the unit cell
    beta : float
        interaxial angle, beta, of the unit cell
    gamma : float
        interaxial angle, gamma, of the unit cell
    sg : str or int
        If int, this should specify the space group number. If str, 
        this should be a space group symbol
    logfile : str or file
        name of a log file to parse to get cell parameters and sg
    """
    return precognition.read(hklfile, a, b, c, alpha, beta, gamma, sg, logfile)

