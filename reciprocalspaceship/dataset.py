import pandas as pd
from pandas.api.types import is_complex_dtype
import numpy as np
import gemmi
import reciprocalspaceship as rs
from reciprocalspaceship.dataseries import DataSeries
from reciprocalspaceship import utils
from reciprocalspaceship.utils import (
    apply_to_hkl,
    phase_shift,
    is_centric,
    is_absent,
    in_asu,
    hkl_to_asu,
    hkl_to_observed,
    compute_dHKL,
    compute_structurefactor_multiplicity,
    bin_by_percentile
)
from functools import wraps
from inspect import signature

def inplace(f):
    """ 
    A decorator that applies the inplace argument. 

    Base function must have a bool param called "inplace" in  the call 
    signature. The position of `inplace` argument doesn't matter.
    """
    @wraps(f)
    def wrapped(ds, *args, **kwargs):
        sig = signature(f)
        bargs = sig.bind(ds, *args, **kwargs)
        bargs.apply_defaults()
        if 'inplace' in bargs.arguments:
            if bargs.arguments['inplace']:
                return f(ds, *args, **kwargs)
            else:
                return f(ds.copy(), *args, **kwargs)
        else:
            raise KeyError(f'"inplace" not found in local variables of @inplacemethod decorated function {f} '
                             "Edit your method definition to include `inplace=Bool`. "
                             )
    return wrapped

def range_indexed(f):
    """
    A decorator that presents the calling dataset with a range index.

    This decorator facilitates writing methods that are agnostic to the 
    true indices in a DataSet. Original index columns are preserved through 
    wrapped function calls.
    """
    @wraps(f)
    def wrapped(ds, *args, **kwargs):
        names = ds.index.names
        ds = ds._index_from_names([None], inplace=True)
        result = f(ds, *args, **kwargs)
        result = result._index_from_names(names, inplace=True)
        ds = ds._index_from_names(names, inplace=True)
        return result.__finalize__(ds)
    return wrapped

class DataSet(pd.DataFrame):
    """
    Representation of a crystallographic dataset.

    A DataSet object provides a tabular representation of reflection data. 
    Reflections are conventionally indexed by Miller indices (rows), but 
    can also be indexed by additional metadata. Per-reflection data can be 
    stored as columns. For additional information about inherited methods 
    and attributes, please see the `Pandas.DataFrame documentation`_.
    
    .. _Pandas.DataFrame documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    """
    _metadata = ['_spacegroup', '_cell', '_index_dtypes', '_merged']

    #-------------------------------------------------------------------
    # __init__ method
    
    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, spacegroup=None, cell=None, merged=None):
        self._index_dtypes = {}
        self._spacegroup = None
        self._cell = None
        self._merged = None
        
        if isinstance(data, DataSet):
            self.__finalize__(data)

        elif isinstance(data, gemmi.Mtz):
            from reciprocalspaceship import io
            dataset = io.from_gemmi(data)
            self.__finalize__(dataset)
            data = dataset

        # Provided values for DataSet attributes take precedence
        if spacegroup:
            self._spacegroup = spacegroup
        if cell:
            self._cell = cell
        if merged is not None:
            self._merged = merged
            
        # Build DataSet using DataFrame.__init__()
        super().__init__(data=data, index=index, columns=columns,
                         dtype=dtype, copy=copy)
        return

    #-------------------------------------------------------------------
    # Attributes
    
    @property
    def _constructor(self):
        return DataSet
    
    @property
    def _constructor_sliced(self):
        return DataSeries

    @property
    def spacegroup(self):
        """Crystallographic space group"""
        return self._spacegroup

    @spacegroup.setter
    def spacegroup(self, val):
        # GH#18: Type-checking for supported input types
        if isinstance(val, gemmi.SpaceGroup) or (val is None):
            self._spacegroup = val
        elif isinstance(val, (str, int)):
            self._spacegroup = gemmi.SpaceGroup(val)
        else:
            raise ValueError(f"Cannot construct gemmi.SpaceGroup from value: {val}")
            
    @property
    def cell(self):
        """Unit cell parameters (a, b, c, alpha, beta, gamma)"""
        return self._cell

    @cell.setter
    def cell(self, val):
        # GH#18: Type-checking for supported input types
        if isinstance(val, gemmi.UnitCell) or (val is None):
            self._cell = val
        elif isinstance(val, (list, tuple, np.ndarray)) and len(val) == 6:
            self._cell = gemmi.UnitCell(*val)
        else:
            raise ValueError(f"Cannot construct gemmi.UnitCell from value: {val}")
            
            
    @property
    def merged(self):
        """Whether DataSet contains merged reflection data (boolean)"""
        return self._merged

    @merged.setter
    def merged(self, val):
        self._merged = val

    #-------------------------------------------------------------------
    # Methods

    @inplace
    def _index_from_names(self, names, inplace=False):
        """Helper method for decorators"""
        if names == [None] and self.index.names != [None]:
            self.reset_index(inplace=True)
        elif names != [None] and self.index.names == [None]:
            self.set_index(names, inplace=True)
        elif names != self.index.names:
            self.reset_index(inplace=True)
            self.set_index(names, inplace=True)
        return self

    def set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False):
        """
        Set the DataSet index using existing columns.

        Set the DataSet index (row labels) using one or more existing 
        columns or arrays (of the correct length). The index can replace
        the existing index or expand on it.

        Parameters
        ----------
        keys : label or array-like or list of labels/arrays
            This parameter can be either a single column key, a single 
            array of the same length as the calling DataSet, or a list 
            containing an arbitrary combination of column keys and arrays. 
        drop : bool
            Whether to delete columns to be used as the new index.
        append : bool
            Whether to append columns to existing index.
        inplace  : bool
            Modify the DataFrame in place (do not create a new object).
        verify_integrity : bool
            Check the new index for duplicates. Otherwise defer the check
            until necessary. Setting to False will improve the performance 
            of this method

        Returns
        -------
        DataSet or None
            DataSet with the new index or None if `inplace=True`

        See Also
        --------
        DataSet.reset_index : Reset index
        """
        if not isinstance(keys, list):
            keys = [keys]

        # Copy dtypes of keys to cache
        for key in keys:
            if isinstance(key, str):
                self._index_dtypes[key] = self[key].dtype.name
            elif isinstance(key, (pd.Index, pd.Series)):
                self._index_dtypes[key.name] = key.dtype.name
            elif isinstance(key, (np.ndarray, list)):
                continue
            else:
                raise ValueError(f"{key} is not an instance of type str, np.ndarray, pd.Index, pd.Series, or list")
            
        return super().set_index(keys, drop, append, inplace, verify_integrity)

    def reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=''):
        """
        Reset the index or a specific level of a MultiIndex.

        Reset the index to use a numbered RangeIndex. Using the `level`
        argument, it is possible to reset one or more levels of a MultiIndex.

        Parameters
        ----------
        level : int, str, tuple, list
            Only remove given levels from the index. Defaults to all levels
        drop : bool
            Do not try to insert index into dataframe columns.
        inplace ; bool
            Modify the DataSet in place (do not create a new object).
        col_level : int or str
            If the columns have multiple levels, determines which level 
            the labels are inserted into. By default it is inserted into
            the first level.
        col_fill : object
            If the columns have multiple levels, determines how the other
            levels are named. If None then the index name is repeated.

        Returns
        -------
        DataSet or None
            DataSet with the new index or None if `inplace=True`

        See Also
        --------
        DataSet.set_index : Set index
        """
        
        # GH#6: Handle level argument to reset_index
        columns = level
        if level is None:
            columns = list(self.index.names)
        
        def _handle_cached_dtypes(dataset, columns, drop):
            """Use _index_dtypes to restore dtypes"""
            if drop:
                for key in columns:
                    if key in dataset._index_dtypes:
                        dataset._index_dtypes.pop(key)
            else:
                for key in columns:
                    if key in dataset._index_dtypes:
                        dtype = dataset._index_dtypes.pop(key)
                        dataset[key] = dataset[key].astype(dtype)
            return dataset
        
        if inplace:
            super().reset_index(level, drop, inplace, col_level, col_fill)
            _handle_cached_dtypes(self, columns, drop)
            return
        else:
            dataset = super().reset_index(level, drop, inplace, col_level, col_fill)
            dataset._index_dtypes = dataset._index_dtypes.copy()
            dataset = _handle_cached_dtypes(dataset, columns, drop)
            return dataset

    @classmethod
    def from_gemmi(cls, gemmiMtz):
        """
        Creates DataSet object from gemmi.Mtz object.

        If the gemmi.Mtz object contains an M/ISYM column and contains duplicated
        Miller indices, an unmerged DataSet will be constructed. The Miller indices 
        will be mapped to their observed values, and a partiality flag will be 
        extracted and stored as a boolean column with the label, ``PARTIAL``. 
        Otherwise, a merged DataSet will be constructed.

        If columns are found with the ``MTZInt`` dtype and are labeled ``PARTIAL``
        or ``CENTRIC``, these will be interpreted as boolean flags used to 
        label partial or centric reflections, respectively.

        Parameters
        ----------
        gemmiMtz : gemmi.Mtz

        Returns
        -------
        DataSet
        """
        return cls(gemmiMtz)

    def to_gemmi(self, skip_problem_mtztypes=False):
        """
        Creates gemmi.Mtz object from DataSet object.

        If ``dataset.merged == False``, the reflections will be mapped to the
        reciprocal space ASU, and a M/ISYM column will be constructed. 

        If boolean flags with the label ``PARTIAL`` or ``CENTRIC`` are found
        in the DataSet, these will be cast to the ``MTZInt`` dtype, and included
        in the gemmi.Mtz object. 

        Parameters
        ----------
        skip_problem_mtztypes : bool
            Whether to skip columns in DataSet that do not have specified
            MTZ datatypes

        Returns
        -------
        gemmi.Mtz
        """
        from reciprocalspaceship import io
        return io.to_gemmi(self, skip_problem_mtztypes)

    def to_structurefactor(self, sf_key, phase_key):
        """
        Convert structure factor amplitudes and phases to complex structure
        factors

        Parameters
        ----------
        sf_key : str
            Column label for structure factor amplitudes
        phase_key : str
            Column label for phases

        Returns
        -------
        rs.DataSeries
            Complex structure factors

        See Also
        --------
        DataSet.from_structurefactor : Convert complex structure factor to amplitude and phase
        """
        sfs = utils.to_structurefactor(self[sf_key], self[phase_key])
        return rs.DataSeries(sfs, index=self.index)

    def from_structurefactor(self, sf_key):
        """
        Convert complex structure factors to structure factor amplitudes
        and phases

        Parameters
        ----------
        sf_key : str
            Column label for complex structure factors
        
        Returns
        -------
        (sf, phase) : tuple of DataSeris
            Tuple of DataSeries for the structure factor amplitudes and
            phases corresponding to the complex structure factors

        See Also
        --------
        DataSet.to_structurefactor : Convert amplitude and phase to complex structure factor
        """
        return utils.from_structurefactor(self[sf_key])
        
    def append(self, *args, check_isomorphous=True, **kwargs):
        """
        Append rows of `other` to the end of calling DataSet, returning
        a new DataSet object. Any columns in `other` that are not present
        in the calling DataSet are added as new columns. 

        For additional documentation on accepted arguments, see the 
        `Pandas DataFrame.append() API <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html>`_.

        Parameters
        ----------
        check_isomorphous : bool
            If True, the spacegroup and cell attributes of DataSets in `other`
            will be compared to those of the calling DataSet to ensure
            they are isomorphous. 

        Returns
        -------
        rs.DataSet

        See Also
        --------
        concat : Concatenate ``rs`` objects
        """
        other = kwargs.get("other", args[0])
        if check_isomorphous:
            if isinstance(other, (list, tuple)):
                for o in other:
                    if isinstance(o, DataSet) and not self.is_isomorphous(o):
                        raise ValueError("DataSet in `other` is not isomorphous")
            else:
                if isinstance(other, DataSet) and not self.is_isomorphous(other):
                    raise ValueError("`other` DataSet is not isomorphous")
        result = super().append(*args, **kwargs)

        # If `ignore_index=True`, the _index_dtypes attribute should
        # be reset.
        if isinstance(result.index, pd.RangeIndex) and self._index_dtypes != {}:
            result.__finalize__(self)
            result._index_dtypes = {}
            return result
        
        return result.__finalize__(self)

    def merge(self, *args, check_isomorphous=True, **kwargs):
        """
        Merge DataSet or named DataSeries using a database-style join on
        columns or indices.

        For additional documentation on accepted arguments, see the 
        `Pandas DataFrame.merge() API <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html>`_.

        Parameters
        ----------
        check_isomorphous : bool
            If True, the spacegroup and cell attributes of DataSets in `other`
            will be compared to those of the calling DataSet to ensure
            they are isomorphous. 

        Returns
        -------
        rs.DataSet

        See Also
        --------
        DataSet.join : Similar method with support for lists of ``rs`` objects
        """
        right = kwargs.get("right", args[0])
        if check_isomorphous and isinstance(right, DataSet):
            if not self.is_isomorphous(right):
                raise ValueError("`right` DataSet is not isomorphous")
        result = super().merge(*args, **kwargs)
        return result.__finalize__(self)

    def join(self, *args, check_isomorphous=True, **kwargs):
        """
        Join DataSets or named DataSeries using a database-style join on
        columns or indices. This method can be used to join lists ``rs`` objects
        to a given DataSet. 

        For additional documentation on accepted arguments, see the 
        `Pandas DataFrame.join() API <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html>`_.

        Parameters
        ----------
        check_isomorphous : bool
            If True, the spacegroup and cell attributes of DataSets in `other`
            will be compared to those of the calling DataSet to ensure
            they are isomorphous. 

        Returns
        -------
        rs.DataSet

        See Also
        --------
        DataSet.merge : Similar method with added flexibility for distinct column labels
        """
        other = kwargs.get("other", args[0])
        if check_isomorphous:
            if isinstance(other, (list, tuple)):
                for o in other:
                    if isinstance(o, DataSet) and not self.is_isomorphous(o):
                        raise ValueError("DataSet in `other` is not isomorphous")
            else:
                if isinstance(other, DataSet) and not self.is_isomorphous(other):
                    raise ValueError("`other` DataSet is not isomorphous")
        result = super().join(*args, **kwargs)
        return result.__finalize__(self)
    
    def write_mtz(self, mtzfile, skip_problem_mtztypes=False):
        """
        Write DataSet to MTZ file.

        If ``DataSet.merged == False``, the reflections will be mapped to the
        reciprocal space ASU, and a M/ISYM column will be constructed. 
        
        If boolean flags with the label ``PARTIAL`` or ``CENTRIC`` are found
        in the DataSet, these will be cast to the ``MTZInt`` dtype, and included
        in the output MTZ file. 

        Parameters
        ----------
        mtzfile : str or file
            name of an mtz file or a file object
        skip_problem_mtztypes : bool
            Whether to skip columns in DataSet that do not have specified
            MTZ datatypes
        """
        from reciprocalspaceship import io
        return io.write_mtz(self, mtzfile, skip_problem_mtztypes)

    def get_phase_keys(self):
        """
        Return column labels for data with Phase dtype.

        Returns
        -------
        keys : list of strings
            list of column labels with ``Phase`` dtype
        """
        keys = [ k for k in self if isinstance(self.dtypes[k], rs.PhaseDtype) ]
        return keys

    def get_complex_keys(self):
        """
        Return columns labels for data with complex dtype.

        Returns
        -------
        keys : list of strings
            list of column labels with complex dtype
        """
        keys = [ k for k in self if is_complex_dtype(self.dtypes[k]) ]
        return keys

    def get_m_isym_keys(self):
        """
        Return column labels for data with M/ISYM dtype.

        Returns
        -------
        key : list of strings
            list of column labels with ``M/ISYM`` dtype
        """
        keys = [ k for k in self if isinstance(self.dtypes[k], rs.M_IsymDtype) ]
        return keys

    @inplace
    @range_indexed
    def apply_symop(self, symop, inplace=False):
        """
        Apply symmetry operation to all reflections in DataSet object. 

        Parameters
        ----------
        symop : str, gemmi.Op
            Gemmi symmetry operation or string representing symmetry op
        inplace : bool
            Whether to return a new DataFrame or make the change in place
        """
        if isinstance(symop, str):
            symop = gemmi.Op(symop)
        elif not isinstance(symop, gemmi.Op):
            raise ValueError(f"Provided symop is not of type gemmi.Op")

        dataset = self

        # Handle phase flips associated with Friedel operator
        if symop.det_rot() < 0:
            phic = -1
        else:
            phic = 1

        # Apply symop to generate new HKL indices and phase shifts
        H = dataset.get_hkls()
        hkl = apply_to_hkl(H, symop)
        phase_shifts = phase_shift(H, symop)
            
        dataset[['H', 'K', 'L']] = hkl
        dataset[['H', 'K', 'L']] = dataset[['H', 'K', 'L']].astype(rs.HKLIndexDtype())

        # Shift phases according to symop
        for key in dataset.get_phase_keys():
            dataset[key] += np.rad2deg(phase_shifts)
            dataset[key] *= phic
            dataset[key] = utils.canonicalize_phases(dataset[key], deg=True)
        for key in dataset.get_complex_keys():
            dataset[key] *= np.exp(1j*phase_shifts)
            if symop.det_rot() < 0:
                dataset[key] = np.conjugate(dataset[key])
        
        return dataset

    def get_hkls(self):
        """
        Get the Miller indices in the DataSet as a ndarray.

        Returns
        -------
        hkl : ndarray, shape=(n_reflections, 3)
            Miller indices in DataSet 
        """
        hkl = self.reset_index()[['H', 'K', 'L']].to_numpy(dtype=np.int32)
        return hkl

    @inplace
    def label_centrics(self, inplace=False):
        """
        Label centric reflections in DataSet. A new column of
        booleans, "CENTRIC", is added to the object.

        Parameters
        ----------
        inplace : bool
            Whether to add the column in place or to return a copy
        """
        self['CENTRIC'] = is_centric(self.get_hkls(), self.spacegroup)
        return self

    @inplace
    def label_absences(self, inplace=False):
        """
        Label systematically absent reflections in DataSet. A new column 
        of booleans, "ABSENT", is added to the object.

        Parameters
        ----------
        inplace : bool
            Whether to add the column in place or to return a copy
        """
        self['ABSENT'] = is_absent(self.get_hkls(), self.spacegroup)
        return self

    @inplace
    def infer_mtz_dtypes(self, inplace=False, index=True):
        """
        Infers MTZ dtypes from column names and underlying data. This 
        method iterates over each column in the DataSet and tries to infer
        its proper MTZ dtype based on common MTZ naming conventions.

        If a given column is already a MTZDtype, its type will be unchanged. 
        If index is True, the MTZ dtypes will be inferred for named columns
        in the index.

        Parameters
        ----------
        inplace : bool
            Whether to modify the dtypes in place or to return a copy
        index : bool
            Infer MTZ dtypes for named column(s) in the DataSet index

        Returns
        -------
        DataSet

        See Also
        --------
        DataSeries.infer_mtz_dtype : Infer MTZ dtype for DataSeries
        """
        # See GH#2: Handle unnamed Index objects such as RangeIndex
        if index:
            index_keys = list(filter(None, self.index.names))
            if index_keys:
                self.reset_index(inplace=True, level=index_keys)

        for c in self:
            self[c] = self[c].infer_mtz_dtype()

        if index and index_keys:
            self.set_index(index_keys, inplace=True)

        return self

    @inplace
    def compute_dHKL(self, inplace=False):
        """
        Compute the real space lattice plane spacing, d, associated with
        the HKL indices in the object.

        Parameters
        ----------
        inplace : bool
            Whether to add the column in place or return a copy
        """
        dHKL = compute_dHKL(self.get_hkls(), self.cell)
        self['dHKL'] = rs.DataSeries(dHKL, dtype='R', index=self.index)
        return self

    @inplace
    def compute_multiplicity(self, inplace=False, include_centering=True):
        """
        Compute the multiplicity of reflections in DataSet. A new column of
        floats, "EPSILON", is added to the object.

        Parameters
        ----------
        inplace : bool
            Whether to add the column in place or to return a copy
        include_centering : bool
            Whether to include centering operations in the multiplicity calculation.
            The default is to include them.
        """
        epsilon = compute_structurefactor_multiplicity(self.get_hkls(), self.spacegroup, include_centering)
        self['EPSILON'] = rs.DataSeries(epsilon, dtype='I', index=self.index)
        return self

    @inplace
    def assign_resolution_bins(self, bins=20, inplace=False, return_labels=True):
        """
        Assign reflections in DataSet to resolution bins.

        Parameters
        ----------
        bins : int
            Number of bins
        inplace : bool
            Whether to add the column in place or return a copy
        return_labels : bool
            Whether to return a list of labels corresponding to the edges
            of each resolution bin

        Returns
        -------
        (DataSet, list) or DataSet
        """
        dHKL = self.compute_dHKL()["dHKL"]

        assignments, labels = bin_by_percentile(dHKL, bins=bins, ascending=False)
        self["bin"] = rs.DataSeries(assignments, dtype="I", index=self.index)

        if return_labels:
            return self, labels
        else:
            return self
        
    def stack_anomalous(self, plus_labels=None, minus_labels=None):
        """
        Convert data from two-column anomalous format to one-column
        format. Intensities, structure factor amplitudes, or other data 
        are converted from separate columns corresponding to a single 
        Miller index to the same data column at different rows indexed 
        by the Friedel-plus or Friedel-minus Miller index. 

        This method will return a DataSet with, at most, twice as many rows as the 
        original --  one row for each Friedel pair. In most cases, the resulting 
        DataSet will be smaller, because centric reflections will not be stacked. 
        For a merged DataSet, this has the effect of mapping reflections 
        from the positive reciprocal space ASU to the positive and negative 
        reciprocal space ASU, for Friedel-plus and Friedel-minus reflections, 
        respectively. 

        Notes
        -----
        - A ValueError is raised if invoked with an unmerged DataSet
        - It is assumed that Friedel-plus column labels are suffixed with (+),
          and that Friedel-minus column labels are suffixed with (-)
        - Corresponding column labels are expected to be given in the same order

        Parameters
        ----------
        plus_labels: str or list-like
            Column label or list of column labels of data associated with
            Friedel-plus reflection (Defaults to columns suffixed with "(+)")
        minus_labels: str or list-like
            Column label or list of column labels of data associated with
            Friedel-minus reflection (Defaults to columns suffixed with "(-)")

        Returns
        -------
        DataSet

        See Also
        --------
        DataSet.unstack_anomalous : Opposite of stack_anomalous
        """
        if not self.merged:
            raise ValueError("DataSet.stack_anomalous() cannot be called with unmerged data")

        # Default behavior: Use labels suffixed with "(+)" or "(-)"
        if (plus_labels is None and minus_labels is None):
            plus_labels  = [ l for l in self.columns if "(+)" in l ]
            minus_labels = [ l for l in self.columns if "(-)" in l ]
        
        # Validate column labels
        if isinstance(plus_labels, str) and isinstance(minus_labels, str):
            plus_labels = [plus_labels]
            minus_labels =[minus_labels]
        elif (isinstance(plus_labels, list) and
              isinstance(minus_labels, list)):
            if len(plus_labels) != len(minus_labels):
                raise ValueError(f"plus_labels: {plus_labels} and minus_labels: "
                                 f"{minus_labels} do not have same length.")
        else:
            raise ValueError(f"plus_labels and minus_labels must have same type "
                             f"and be str or list: plus_labels is type "
                             f"{type(plus_labels)} and minus_labe is type "
                             f"{type(minus_labels)}.")

        for plus, minus in zip(plus_labels, minus_labels):
            if self[plus].dtype != self[minus].dtype:
                raise ValueError(f"Corresponding labels in {plus_labels} and "
                                 f"{minus_labels} are not the same dtype: "
                                 f"{self[plus].dtype} and {self[minus].dtype}")

        # Map Friedel reflections to +/- ASU
        centrics = self.label_centrics()["CENTRIC"]
        dataset_plus = self.drop(columns=minus_labels)
        dataset_minus = self.loc[~centrics].drop(columns=plus_labels)
        dataset_minus.apply_symop(gemmi.Op("-x,-y,-z"), inplace=True)

        # Rename columns and update dtypes
        new_labels = [ l.rstrip("(+)") for l in plus_labels ]
        column_mapping_plus  = dict(zip(plus_labels, new_labels))
        column_mapping_minus = dict(zip(minus_labels, new_labels))
        dataset_plus.rename(columns=column_mapping_plus, inplace=True)
        dataset_minus.rename(columns=column_mapping_minus, inplace=True)
        F = dataset_plus.append(dataset_minus)
        for label in new_labels:
            F[label] = F[label].from_friedel_dtype()
            
        return F

    @range_indexed
    def unstack_anomalous(self, columns=None, suffixes=("(+)", "(-)")):
        """
        Convert data from one-column format to two-column anomalous
        format. Provided column labels are converted from separate rows 
        indexed by their Friedel-plus or Friedel-minus Miller index to 
        different columns indexed at the Friedel-plus HKL.

        This method will return a smaller DataSet than the original -- 
        Friedel pairs will both be indexed at the Friedel-plus index. This 
        has the effect of mapping reflections to the positive reciprocal 
        space ASU, including data for both Friedel pairs at the Friedel-plus 
        Miller index.

        Notes
        -----
        - A ValueError is raised if invoked with an unmerged DataSet

        Parameters
        ----------
        columns : str or list-like
            Column label or list of column labels of data that should be 
            associated with Friedel pairs. If None, all columns are 
            converted to the two-column anomalous format.
        suffixes : tuple or list  of str
            Suffixes to append to Friedel-plus and Friedel-minus data 
            columns

        Returns
        -------
        DataSet

        See Also
        --------
        DataSet.stack_anomalous : Opposite of unstack_anomalous
        """
        if not self.merged:
            raise ValueError("DataSet.unstack_anomalous() cannot be called with unmerged data")

        # Validate input
        if columns is None:
            columns = self.columns.to_list()
        elif isinstance(columns, str):
            columns = [columns]
        elif not isinstance(columns, (list, tuple)):
            raise ValueError(f"Expected columns to be str, list or tuple. "
                             f"Provided value is type {type(columns)}")
        
        if not (isinstance(suffixes, (list, tuple)) and len(suffixes) == 2):
            raise ValueError(f"Expected suffixes to be tuple or list of len() of 2")

        # Separate DataSet into Friedel(+) and Friedel(-)
        columns = set(columns).union(set(["H", "K", "L"]))
        dataset = self.hkl_to_asu()
        if "PARTIAL" in columns: columns.remove("PARTIAL")
        for column in columns:
            dataset[column] = dataset[column].to_friedel_dtype()
        dataset_plus  = dataset.loc[dataset["M/ISYM"]%2 == 1].copy()
        dataset_minus = dataset.loc[dataset["M/ISYM"]%2 == 0].copy()
        dataset_minus = dataset_minus.loc[:, columns]
        result = dataset_plus.merge(dataset_minus, how="outer",
                                    on=["H", "K", "L"],
                                    suffixes=suffixes)

        # Handle centric reflections
        columns = columns.difference(set(["H", "K", "L"]))
        plus_columns  = [ c + suffixes[0] for c in columns ]
        minus_columns = [ c + suffixes[1] for c in columns ]        
        centrics = result.label_centrics()["CENTRIC"]
        result.loc[centrics, minus_columns] = result.loc[centrics, plus_columns].values
        
        if "M/ISYM" not in self.columns and self.merged:
            result.drop(columns="M/ISYM", inplace=True)
            
        return result
        
    def is_isomorphous(self, other, cell_threshold=0.05):
        """
        Determine whether DataSet is isomorphous to another DataSet. This
        method confirms isomorphism by ensuring the spacegroups are equivalent,
        and that the cell parameters are within a specified percentage 
        (see `cell_threshold`). 

        Parameters
        ----------
        other : rs.DataSet
            DataSet to which it will be compared
        cell_threshold : float
            Acceptable percent difference between unit cell parameters

        Returns
        -------
        bool
        """
        # Check for spacegroup and cell attributes
        if not isinstance(self.spacegroup, gemmi.SpaceGroup):
            raise AttributeError(f"Calling dataset's spacegroup attribute is not set")
        elif not isinstance(other.spacegroup, gemmi.SpaceGroup):
            raise AttributeError(f"`other` dataset's spacegroup attribute is not set")
        elif not isinstance(self.cell, gemmi.UnitCell):
            raise AttributeError(f"Calling dataset's cell attribute is not set")
        elif not isinstance(other.cell, gemmi.UnitCell):
            raise AttributeError(f"`other` dataset's cell attribute is not set")            
        
        # Check spacegroup
        if self.spacegroup.xhm() != other.spacegroup.xhm():
            return False

        # Check cell parameters
        params = ["a", "b", "c", "alpha", "beta", "gamma"]
        for param in params:
            param1 = self.cell.__getattribute__(param)
            param2 = other.cell.__getattribute__(param)
            if (np.abs((param1 - param2)) / 100.) > cell_threshold:
                return False
        
        return True

    @inplace
    @range_indexed
    def hkl_to_asu(self, inplace=False, anomalous=False):
        """
        Map HKL indices to the reciprocal space asymmetric unit. If phases
        are included in the DataSet, they will be changed according to the
        phase shift associated with the necessary symmetry operation.

        If ``DataSet.merged == False``, and a partiality flag labeled ``PARTIAL``
        is included in the DataSet, the partiality flag will be used to 
        construct a proper M/ISYM column. Both merged and unmerged DataSets
        will have an M/ISYM column added. 

        Parameters
        ----------
        inplace : bool
            Whether to modify the DataSet in place or return a copy
        anomalous : bool
            If True, acentric reflections will be mapped to the +/- ASU. 
            If False, all reflections are mapped to the Friedel-plus ASU. 

        Returns
        -------
        DataSet

        See Also
        --------
        DataSet.hkl_to_observed : Opposite of DataSet.hkl_to_asu()
        """
        dataset = self

        # Compute new HKLs and phase shifts
        hkls = dataset.get_hkls()
        compressed_hkls, inverse = np.unique(hkls, axis=0, return_inverse=True)
        asu_hkls, isym, phi_coeff, phi_shift = hkl_to_asu(
            compressed_hkls, 
            dataset.spacegroup, 
            return_phase_shifts=True
        )

        dataset[["H", "K", "L"]] = asu_hkls[inverse]
        dataset[["H", "K", "L"]] = dataset[["H", "K", "L"]].astype("HKL")

        # Apply phase shift
        for k in dataset.get_phase_keys():
            dataset[k] = phi_coeff[inverse] * (dataset[k] + phi_shift[inverse])
        dataset.canonicalize_phases(inplace=True)

        # GH#3: if PARTIAL column exists, use it to construct M/ISYM
        if "PARTIAL" in dataset.columns:
            m_isym = isym[inverse] + 256*dataset["PARTIAL"].to_numpy()
            dataset['M/ISYM'] = DataSeries(m_isym, dtype="M/ISYM", index=dataset.index)
            dataset.drop(columns="PARTIAL", inplace=True)
        else:
            dataset['M/ISYM'] = DataSeries(isym[inverse], dtype="M/ISYM", index=dataset.index)

        # GH#16: Handle anomalous flag to separate Friedel pairs
        # GH#25: Centrics should not be considered Friedel pairs
        if anomalous:
            acentric = ~dataset.label_centrics()["CENTRIC"]
            friedel_minus = dataset['M/ISYM']%2 == 0
            dataset[friedel_minus & acentric] = dataset[friedel_minus & acentric].apply_symop("-x,-y,-z")
            
        return dataset

    @inplace
    @range_indexed
    def hkl_to_observed(self, m_isym=None, inplace=False):
        """
        Map HKL indices to their observed index using an ``M/ISYM`` column. 
        This method applies the symmetry operation specified by the ``M/ISYM``
        column to each Miller index in the DataSet. If phases are included
        in the DataSet, they will be changed by the phase shift associated
        with the symmetry operation.

        If ``DataSet.merged == False``, the ``M/ISYM`` column is used to 
        construct a partiality flag labeled ``PARTIAL``. This is added to 
        the DataSet, and the M/ISYM column is dropped. If 
        ``DataSet.merged == True``, the ``M/ISYM`` column is dropped, but
        a partiality flag is not added.

        Parameters
        ----------
        m_isym : str
            Column label for M/ISYM values in DataSet. If m_isym is None 
            and a single M/ISYM column is present, it will automatically 
            be used.
        inplace : bool
            Whether to modify the DataSet in place or return a copy

        Returns
        -------
        DataSet

        See Also
        --------
        DataSet.hkl_to_asu : Opposite of DataSet.hkl_to_observed()
        """ 
        # Validate input
        if m_isym is None:
            m_isym = self.get_m_isym_keys()
            if len(m_isym) == 1:
                m_isym = m_isym[0]
            else:
                raise ValueError(f"Method requires a single M/ISYM column -- found: {m_isym}")
        elif not isinstance(m_isym, str):
            raise ValueError("Provided M/ISYM column label should be type str")
        elif not isinstance(self.dtypes[m_isym], rs.M_IsymDtype):
            raise ValueError(f"Provided M/ISYM column label is of wrong dtype")

        # GH#3: Separate combined M/ISYM into M and ISYM        
        isym = (self[m_isym] % 256).to_numpy(dtype=np.int32)
        if not self.merged:
            self["PARTIAL"] = (self[m_isym]/256).astype(int) != 0
        self.drop(columns=m_isym, inplace=True)
        
        # Compute new HKLs and phase shifts
        hkls = self.get_hkls()
        hkls_isym = np.concatenate([hkls, isym.reshape(-1, 1)], axis=1)
        compressed, inverse = np.unique(hkls_isym, axis=0, return_inverse=True)
        observed_hkls, phi_coeff, phi_shift = hkl_to_observed(
            compressed[:, :3],       # compressed HKLs
            compressed[:, 3],        # compressed ISYM
            self.spacegroup,
            return_phase_shifts=True
        )
        self[["H", "K", "L"]] = observed_hkls[inverse]
        self[["H", "K", "L"]] = self[["H", "K", "L"]].astype("HKL")

        # Apply phase shift
        for k in self.get_phase_keys():
            self[k] = phi_coeff[inverse] * (self[k] + phi_shift[inverse])
        self.canonicalize_phases(inplace=True)
        
        return self

    def expand_to_p1(self):
        """
        Generates all symmetrically equivalent reflections. The spacegroup 
        symmetry is set to P1.
        
        Returns
        -------
        DataSet
        """
        if not self.merged:
            raise ValueError("This function is only applicable for merged DataSets")

        groupops = self.spacegroup.operations()
        p1 = rs.concat([ self.apply_symop(op) for op in groupops ])
        p1.spacegroup = gemmi.SpaceGroup(1)
        p1.reset_index(inplace=True)
        p1.drop_duplicates(subset=["H", "K", "L"], inplace=True)
        p1.set_index(["H", "K", "L"], inplace=True)
        return p1

    def expand_anomalous(self):
        """
        Expands data by applying Friedel operator (-x, -y, -z). The
        necessary phase shifts are made for columns of complex
        dtypes or PhaseDtypes. 
        
        Returns
        -------
        DataSet
        """
        friedel = self.apply_symop("-x,-y,-z")
        return self.append(friedel)
    
    @inplace
    def canonicalize_phases(self, inplace=False):
        """
        Canonicalize columns with phase data to fall in the interval between
        -180 and 180 degrees. This method will modify the values within any
        column composed of data with the PhaseDtype.

        Parameters
        ----------
        inplace : bool
            Whether to modify the DataSet in place or return a copy

        Returns
        -------
        DataSet
        """
        for k in self.get_phase_keys():
            self[k] = utils.canonicalize_phases(self[k])

        return self

    def to_reciprocalgrid(self, key, gridsize, friedels_law=True):
        """
        Set up reciprocal grid with values from column, ``key``, indexed by
        Miller indices. A 3D numpy array will be initialized with the
        specified gridsize. Any missing Miller indices are initialized to
        zero.

        Notes
        -----
        - The data being arranged on a reciprocal grid must be compatible
          with a numpy datatype.
        - If the data is complex and ``friedels_law=True``, the complex
          conjugate will be used to populate Friedel pairs. The only scenario
          in which to use ``friedels_law=False`` would be if Friedel pairs
          have already been stacked to go from two-column to one-column
          anomalous data. 
        
        Parameters
        ----------
        key : str
            Column label for value to arrange on reciprocal grid
        gridsize : array-like (len==3)
            Dimensions for 3D reciprocal grid.
        friedels_law : bool
            Whether to expand data from P1 ASU to P1 unit cell when constructing
            reciprocal grid. If False, Friedel pairs must be handled by
            the user

        Returns
        -------
        numpy.ndarray
        """
        p1 = self.expand_to_p1()
        if friedels_law:
            p1 = p1.expand_anomalous()
        p1.sort_index(inplace=True)
        data = p1[key].to_numpy()
        H = p1.get_hkls()
        grid = np.zeros(gridsize, dtype=data.dtype)
        grid[H[:, 0], H[:, 1], H[:, 2]] = data
        return grid
