from .formats import hkl
import pandas as pd

class CrystalSeries(pd.Series):
    """
    Representation of a sliced Crystal
    """
    spacegroup = None
    cell = None

    @property
    def _constructor(self):
        return CrystalSeries

class Crystal(pd.DataFrame):
    """
    Representation of a crystal

    Attributes
    ----------
    spacegroup : int
        Number corresponding to the crystal space group
    cell : np.ndarray
        Unit cell constants of crystal (a, b, c, alpha, beta, gamma)
    """

    _metadata = ['spacegroup', 'cell']
    spacegroup = None
    cell = None

    @property
    def _constructor(self):
        return Crystal

    @property
    def _constructor_sliced(self):
        return CrystalSeries

    def read_hkl(self, hklfile, a=None, b=None, c=None, alpha=None,
                 beta=None, gamma=None, sg=None):
        """
        Initialize attributes and populate the crystal object with data 
        from a HKL file of reflections. This is the output format used 
        by Precognition when processing Laue diffraction data.

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
        """
        return hkl.read(self, hklfile, a, b, c, alpha, beta, gamma, sg)
