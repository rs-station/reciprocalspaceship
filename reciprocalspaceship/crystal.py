import pandas as pd

class CrystalSeries(pd.Series):
    """
    Representation of a sliced Crystal
    """

    _metadata = ["mtztype"]
    
    def __init__(self, *args, **kwargs):
        self.mtztype = kwargs.pop('mtztype', None)
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
    spacegroup : gemmi.SpaceGroup
        Crystallographic space group containing symmetry operations
    cell : gemmi.UnitCell
        Unit cell constants of crystal (a, b, c, alpha, beta, gamma)
    """

    _metadata = ['spacegroup', 'cell', 'mtztype']

    def __init__(self, *args, **kwargs):
        self.spacegroup = kwargs.pop('spacegroup', None)
        self.cell = kwargs.pop('cell', None)
        super().__init__(*args, **kwargs)
        return
    
    @property
    def _constructor(self):
        return Crystal

    @property
    def _constructor_sliced(self):
        return CrystalSeries
    
