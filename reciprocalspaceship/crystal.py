import pandas as pd
import numpy as np

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

    def get_phase_keys(self):
        """
        Return column labels associated with phase data

        Returns
        -------
        keys : list of strings
            list of column labels
        """
        keys = []
        for k in self:
            try:
                if self[k].dtype.mtztype == "P":
                    keys.append(k)
            except:
                continue
        return keys
    
    def _label_centrics(self):
        """
        Label centric reflections in Crystal object. A new column of
        booleans, "CENTRIC", is added to the object.
        """
        self['CENTRIC'] = False
        hkl = np.vstack(self.index)
        for op in self.spacegroup.operations():

            # TODO: This for loop could be removed if op.apply_to_hkl()
            #       were vectorized
            newhkl = hkl.copy()
            for i, h in enumerate(hkl):
                newhkl[i] = op.apply_to_hkl(h)
                
            self['CENTRIC'] = np.all(np.isclose(newhkl, -hkl), 1) | self['CENTRIC']
        return self

    def unmerge_anomalous(self, inplace=False):
        """
        Unmerge Friedel pairs. In the near future, this should probably
        be rolled into a bigger unmerge() function
        """
        self._label_centrics()
        Fplus = self.copy()
        Fminus = self.copy().reset_index()
        Fminus[['H', 'K', 'L']] = -1*Fminus[['H', 'K', 'L']]
        for k in self.get_phase_keys():
            Fminus.loc[~Fminus.CENTRIC, k] = -Fminus.loc[~Fminus.CENTRIC, k]
        Fminus = Fminus.set_index(['H', 'K', 'L'])

        F = Fplus.append(Fminus.loc[Fminus.index.difference(Fplus.index)])
        
        if inplace:
            self._data = F._data
            return self
        else:
            return F.__finalize__(self)
