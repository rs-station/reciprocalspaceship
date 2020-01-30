import pandas as pd
import numpy as np
import gemmi
from .utils import canonicalize_phases, apply_to_hkl, phase_shift
from .utils.asu import in_asu,hkl_to_asu
from .dtypes.mapping import mtzcode2dtype

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

    def write_mtz(self, *args, **kwargs):
        """
        Write an MTZ reflection file from the reflection data in a Crystal.

        Parameters
        ----------
        mtzfile : str or file
            name of an mtz file or a file object
        skip_problem_mtztypes : bool
            Whether to skip columns in Crystal that do not have specified
            mtz datatypes
        """
        from reciprocalspaceship import io
        return io.write_mtz(self, *args, **kwargs)

    def write_hkl(self, *args, **kwargs):
        """
        Write contents of Crystal object to an HKL file

        Parameters
        ----------
        outfile : str or file
            name of an hkl file or file-like object
        sf_key : str
            key for structure factor in DataFrame
        err_key : str
            key for structure factor error in DataFrame
        """
        from reciprocalspaceship import io
        return io.write_hkl(self, *args, **kwargs)

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

    def apply_symop(self, symop, inplace=False):
        """
        Apply symmetry operation to all reflections in Crystal object. 

        Parameters
        ----------
        symop : gemmi.Op
            Gemmi symmetry operation
        inplace : bool
            Whether to return a new DataFrame or make the change in place
        """
        if not isinstance(symop, gemmi.Op):
            raise ValueError(f"Provided symop is not of type gemmi.Op")
        
        # Apply symop to generate new HKL indices and phase shifts
        hkl = np.vstack(self.index)
        phase_shifts  = np.zeros(len(hkl))

        hkl = apply_to_hkl(hkl, symop)
        phase_shifts = np.rad2deg(phase_shift(hkl, symop))
            
        if inplace:
            self.reset_index(inplace=True)
            self[['H', 'K', 'L']] = hkl
            self.set_index(['H', 'K', 'L'], inplace=True)
            F = self
        else:
            F = self.copy().reset_index()
            F[['H', 'K', 'L']] = hkl
            F.set_index(['H', 'K', 'L'], inplace=True)

        # Shift phases according to symop
        for key in F.get_phase_keys():
            F[key] += phase_shifts
            F[key] = canonicalize_phases(F[key], deg=True)
            
        return F.__finalize__(self)

    def get_hkls(self):
        hkl = self.reset_index()[['H', 'K', 'L']].to_numpy()
        return hkl

    def label_centrics(self, inplace=False):
        """
        Label centric reflections in Crystal object. A new column of
        booleans, "CENTRIC", is added to the object.

        Parameters
        ----------
        inplace : bool
            Whether to add the column in place or to return a copy
        """
        if inplace:
            crystal = self
        else:
            crystal = self.copy()

        uncompressed_hkls = crystal.get_hkls()
        hkl,inverse = np.unique(uncompressed_hkls, axis=0, return_inverse=True)
        centric = np.zeros(len(hkl), dtype=bool)
        for op in crystal.spacegroup.operations():
            newhkl = apply_to_hkl(hkl, op)
            centric = np.all(newhkl == -hkl, 1) | centric

        crystal['CENTRIC'] = centric[inverse]
        return crystal

    def compute_dHKL(self, inplace=False):
        """
        Compute the real space lattice plane spacing, d, associated with
        the HKL indices in the object.

        Parameters
        ----------
        inplace : bool
            Whether to add the column in place or return a copy
        """
        if inplace:
            crystal = self
        else:
            crystal = self.copy()

        uncompressed_hkls = crystal.get_hkls()
        hkls,inverse = np.unique(uncompressed_hkls, axis=0, return_inverse=True)
        dhkls = np.zeros(len(hkls))
        for i, hkl in enumerate(hkls):
            dhkls[i] = crystal.cell.calculate_d(hkl)
        crystal['dHKL'] = CrystalSeries(dhkls[inverse], dtype="MTZReal")
        return crystal

    def unmerge_anomalous(self, inplace=False):
        """
        Unmerge Friedel pairs. In the near future, this should probably
        be rolled into a bigger unmerge() function
        """
        self.label_centrics(inplace=True)
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

    def hkl_to_asu(self, inplace=False):
        """
        Map all HKL indices to the reciprocal space asymmetric unit; return a copy
        This is provisional. Doesn't quite work yet. 
        """
        if inplace:
            crystal = self
        else:
            crystal = self.copy()

        index_keys = crystal.index.names
        crystal.reset_index(inplace=True)
        hkls = crystal[['H', 'K', 'L']].to_numpy()
        compressed_hkls,inverse = np.unique(hkls, axis=0, return_inverse=True)
        compressed_hkls, isym, phi_coeff, phi_shift = hkl_to_asu(
            compressed_hkls, 
            crystal.spacegroup, 
            return_phase_shifts=True
        )
        crystal.loc[:, ['H', 'K', 'L']] = compressed_hkls[inverse]
        for k in crystal.get_phase_keys():
            crystal[k] = phi_coeff[inverse] * (crystal[k] + phi_shift[inverse])
        crystal['M/ISYM'] = CrystalSeries(isym[inverse], dtype="M_Isym")
        crystal._canonicalize_phases(inplace=True)
        crystal.set_index(['H', 'K', 'L'], inplace=True)
        return crystal

    def _canonicalize_phases(self, inplace=False):
        if inplace:
            new_crystal = self

        else:
            new_crystal = self.copy()

        for k in new_crystal.get_phase_keys():
            new_crystal[k] = canonicalize_phases(new_crystal[k])

        return new_crystal


