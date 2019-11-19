import pandas as pd

class CrystalSeries(pd.Series):
    """
    Representation of a sliced Crystal
    """
    
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
    cache_index_dtypes : Dictionary
        Dictionary of dtypes for columns in Crystal.index. Populated upon
        calls to Crystal.set_index() and depopulated by calls to 
        Crystal.reset_index()
    """

    _metadata = ['spacegroup', 'cell', 'cache_index_dtypes']

    def __init__(self, *args, **kwargs):
        self.spacegroup = kwargs.pop('spacegroup', None)
        self.cell = kwargs.pop('cell', None)
        self.cache_index_dtypes = kwargs.pop('cache_index_dtypes', {})
        super().__init__(*args, **kwargs)
        return
    
    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return Crystal(*args, **kwargs).__finalize__(self)
        return _c

    @property
    def _constructor_sliced(self):
        return CrystalSeries

    def set_index(self, keys, **kwargs):
        
        # Copy dtypes of keys to cache
        for key in keys:
            self.cache_index_dtypes[key] = self[key].dtype.name

        return super().set_index(keys, **kwargs)

    def reset_index(self, **kwargs):
        
        if kwargs.get("inplace", False):
            super().reset_index(**kwargs)

            # Cast all values to cached dtypes
            if not kwargs.get("drop", False):
                for key in self.cache_index_dtypes.keys():
                    dtype = self.cache_index_dtypes[key]
                    self[key] = self[key].astype(dtype)
                self.cache_index_dtypes = {}
            return
        
        else:
            newdf = super().reset_index(**kwargs)

            # Cast all values to cached dtypes
            if not kwargs.get("drop", False):
                for key in newdf.cache_index_dtypes.keys():
                    dtype = newdf.cache_index_dtypes[key]
                    newdf[key] = newdf[key].astype(dtype)
                newdf.cache_index_dtypes = {}
            return newdf

    def write_mtz(self, mtzfile):
        """
        Write an MTZ reflection file from the reflection data in a Crystal.

        Parameters
        ----------
        mtzfile : str or file
            name of an mtz file or a file object
        """
        from reciprocalspaceship import io
        return io.write_mtz(self, mtzfile)

    def _label_centrics(self):
        """
        Add 'CENTRIC' key to self. Label centric reflections as True.
        """
        self['CENTRIC'] = False
        hkl = np.vstack(self.index)
        for op in self.spacegroup.operations:
            newhkl = hkl.copy()
            for i, h in enumerate(hkl):
                newhkl[i] = op.apply_to_hkl(h)
            self['CENTRIC'] = np.all(np.isclose(newhkl, -hkl), 1) | self['CENTRIC']
        return self
