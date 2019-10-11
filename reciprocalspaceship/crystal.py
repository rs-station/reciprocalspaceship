import pandas as pd

class CrystalSeries(pd.Series):
    spacegroup = None
    cell = None
    A = None
    F = None
    V = None

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
