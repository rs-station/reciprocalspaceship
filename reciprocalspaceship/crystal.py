import pandas as pd

class CrystalSeries(pd.Series):
    """
    Representation of a sliced Crystal
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return
    
    @property
    def _constructor(self):
        return CrystalSeries

    @property
    def _constructor_expanddim(self):
        return Crystal
    
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spacegroup = None
        self.cell = None
        return
    
    @property
    def _constructor(self):
        return Crystal

    @property
    def _constructor_sliced(self):
        return CrystalSeries
