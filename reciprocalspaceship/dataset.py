import pandas as pd
import numpy as np
import gemmi
import reciprocalspaceship as rs
from reciprocalspaceship.dataseries import DataSeries
from reciprocalspaceship import utils
from reciprocalspaceship.utils import (
    apply_to_hkl,
    phase_shift,
    is_centric,
    in_asu,
    hkl_to_asu,
    hkl_to_observed,
    compute_dHKL,
)

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
    _metadata = ['_spacegroup', '_cell', '_cache_index_dtypes', '_merged']

    #-------------------------------------------------------------------
    # __init__ method
    
    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, spacegroup=None, cell=None, merged=None):
        self._cache_index_dtypes = {}
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
        if merged:
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
        self._spacegroup = val
    
    @property
    def cell(self):
        """Unit cell parameters (a, b, c, alpha, beta, gamma)"""
        return self._cell

    @cell.setter
    def cell(self, val):
        self._cell = val

    @property
    def merged(self):
        """Whether DataSet contains merged reflection data (boolean)"""
        return self._merged

    @merged.setter
    def merged(self, val):
        self._merged = val

    #-------------------------------------------------------------------
    # Methods
    
    def set_index(self, keys, **kwargs):
        
        # Copy dtypes of keys to cache
        for key in keys:
            self._cache_index_dtypes[key] = self[key].dtype.name

        return super().set_index(keys, **kwargs)

    def reset_index(self, **kwargs):
        
        if kwargs.get("inplace", False):
            super().reset_index(**kwargs)

            # Cast all values to cached dtypes
            if not kwargs.get("drop", False):
                for key in self._cache_index_dtypes.keys():
                    dtype = self._cache_index_dtypes[key]
                    self[key] = self[key].astype(dtype)
                self._cache_index_dtypes = {}
            return
        
        else:
            newdf = super().reset_index(**kwargs)

            # Cast all values to cached dtypes
            if not kwargs.get("drop", False):
                for key in newdf._cache_index_dtypes.keys():
                    dtype = newdf._cache_index_dtypes[key]
                    newdf[key] = newdf[key].astype(dtype)
                newdf._cache_index_dtypes = {}
            return newdf

    @classmethod
    def from_gemmi(cls, gemmiMtz):
        """
        Creates DataSet object from gemmi.Mtz object.

        If the gemmi.Mtz object contains an M/ISYM column, an unmerged
        DataSet will be constructed. The Miller indices will be mapped to
        their observed values, and a partiality flag will be extracted 
        and stored as a boolean column with the label, ``PARTIAL``. 

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
    
    def apply_symop(self, symop, inplace=False):
        """
        Apply symmetry operation to all reflections in DataSet object. 

        Parameters
        ----------
        symop : gemmi.Op
            Gemmi symmetry operation
        inplace : bool
            Whether to return a new DataFrame or make the change in place
        """
        if not isinstance(symop, gemmi.Op):
            raise ValueError(f"Provided symop is not of type gemmi.Op")

        if inplace:
            F = self
        else:
            F = self.copy()
            
        # Apply symop to generate new HKL indices and phase shifts
        H = F.get_hkls()
        hkl = apply_to_hkl(H, symop)
        phase_shifts = np.rad2deg(phase_shift(H, symop))
            
        F.reset_index(inplace=True)
        F[['H', 'K', 'L']] = hkl
        F[['H', 'K', 'L']] = F[['H', 'K', 'L']].astype(rs.HKLIndexDtype())
        F.set_index(['H', 'K', 'L'], inplace=True)

        # Shift phases according to symop
        for key in F.get_phase_keys():
            F[key] += phase_shifts
            F[key] = utils.canonicalize_phases(F[key], deg=True)
            
        return F.__finalize__(self)

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

    def label_centrics(self, inplace=False):
        """
        Label centric reflections in DataSet. A new column of
        booleans, "CENTRIC", is added to the object.

        Parameters
        ----------
        inplace : bool
            Whether to add the column in place or to return a copy
        """
        if inplace:
            dataset = self
        else:
            dataset = self.copy()

        dataset['CENTRIC'] = is_centric(dataset.get_hkls(), dataset.spacegroup)
        return dataset

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
        if inplace:
            dataset = self
        else:
            dataset = self.copy()

        # See GH#2: Handle unnamed Index objects such as RangeIndex
        if index:
            index_keys = list(filter(None, dataset.index.names))
            if index_keys:
                dataset.reset_index(inplace=True, level=index_keys)

        for c in dataset:
            dataset[c] = dataset[c].infer_mtz_dtype()

        if index and index_keys:
            dataset.set_index(index_keys, inplace=True)

        return dataset

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
            dataset = self
        else:
            dataset = self.copy()

        dHKL = compute_dHKL(self.get_hkls(), self.cell)
        dataset['dHKL'] = rs.DataSeries(dHKL, dtype='R', index=dataset.index)
        return dataset

    def stack_anomalous(self, plus_labels=None, minus_labels=None):
        """
        Convert data from two-column anomalous format to one-column
        format. Intensities, structure factor amplitudes, or other data 
        are converted from separate columns corresponding to a single 
        Miller index to the same data column at different rows indexed 
        by the Friedel-plus or Friedel-minus Miller index. 

        If ``DataSet.merged == True``, this method will return a DataSet
        with twice as many rows as the original --  one row for each Friedel
        pair. If ``DataSet.merged == False``, this method will return a 
        DataSet with the same number of rows as the original. 

        For a merged DataSet, this has the effect of mapping reflections 
        from the positive reciprocal space ASU to the positive and negative 
        reciprocal space ASU, for Friedel-plus and Friedel-minus reflections, 
        respectively. For an unmerged DataSet, Bijvoet reflections are mapped
        from the positive reciprocal space ASU to their observed Miller 
        indices. 

        Notes
        -----
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
        if (plus_labels is None and minus_labels is None):
            plus_labels  = [ l for l in self.columns if "(+)" in l ]
            minus_labels = [ l for l in self.columns if "(-)" in l ]
        
        # Check input data
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

        new_labels = [ l.rstrip("(+)") for l in plus_labels ]
        column_mapping_plus  = dict(zip(plus_labels, new_labels))
        column_mapping_minus = dict(zip(minus_labels, new_labels))

        # Handle merged DataSet case
        if self.merged:
            dataset_plus = self.copy()
            dataset_plus.drop(columns=minus_labels, inplace=True)
            dataset_minus = self.copy().drop(columns=plus_labels)
            dataset_minus.apply_symop(gemmi.Op("-x,-y,-z"), inplace=True)
            
        # Handle unmerged DataSet case
        else:
            dataset_plus = self.loc[self[minus_labels].isna().agg("all", axis=1)].copy()
            dataset_minus = self.loc[self[plus_labels].isna().agg("all", axis=1)].copy()
            dataset_plus.drop(columns=minus_labels, inplace=True)
            dataset_minus.drop(columns=plus_labels, inplace=True)
            dataset_plus.hkl_to_observed(inplace=True)
            dataset_minus.hkl_to_observed(inplace=True)

        dataset_plus.rename(columns=column_mapping_plus, inplace=True)
        dataset_minus.rename(columns=column_mapping_minus, inplace=True)
        F = dataset_plus.append(dataset_minus)
        for label in new_labels:
            F[label] = F[label].from_friedel_dtype()
            
        return F.__finalize__(self)

    def unstack_anomalous(self, columns=None, suffixes=("(+)", "(-)")):
        """
        Convert data from one-column format to two-column anomalous
        format. Provided column labels are converted from separate rows 
        indexed by their Friedel-plus or Friedel-minus Miller index to 
        different columns indexed at the Friedel-plus HKL.

        If ``DataSet.merged == True``, this method will return a DataSet
        with half as many rows as the original --  one row with both Friedel
        pairs. If ``DataSet.merged == False``, this method will return a 
        DataSet with the same number of rows as the original. 

        For a merged DataSet, this has the effect of mapping reflections 
        to the positive reciprocal space ASU, including data for both Friedel 
        pairs at the Friedel-plus Miller index. For an unmerged DataSet, 
        all Bijvoet reflections are mapped to the Miller index of the positive 
        reciprocal space ASU; however, the reflection data is kept in distinct
        rows with the anomalous data in columns suffixed according to whether
        the underlying reflection is Friedel-plus or Friedel-minus. A M/ISYM
        column is added to an unmerged DataSet to allow reflections to be
        mapped back to their observed Miller index. 

        Parameters
        ----------
        columns : str or list-like
            Column label or list of column labels of data that should be 
            associated with Friedel pairs. If None, all columns are 
            converted are converted to the two-column anomalous format.
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
        # Validate input
        if columns is None:
            columns = self.columns.to_list()
        elif isinstance(columns, str):
            columns =  [columns]
        elif not isinstance(columns, (list, tuple)):
            raise ValueError(f"Expected columns to be str, list or tuple. "
                             f"Provided value is type {type(columns)}")
            
        if not (isinstance(suffixes, (list, tuple)) and len(suffixes) == 2):
            raise ValueError(f"Expected suffixes to be tuple or list of len() of 2")

        # Separate DataSet into Friedel(+) and Friedel(-)
        dataset = self.hkl_to_asu()
        if "PARTIAL" in columns: columns.remove("PARTIAL")
        for column in columns:
            dataset[column] = dataset[column].to_friedel_dtype()
        dataset_plus  = dataset.loc[dataset["M/ISYM"]%2 == 1].copy()
        dataset_minus = dataset.loc[dataset["M/ISYM"]%2 == 0].copy()

        # Handle merged DataSet
        if self.merged:
            dataset_minus = dataset_minus.loc[:, columns]
            result = dataset_plus.merge(dataset_minus, how="outer",
                                        left_index=True, right_index=True,
                                        suffixes=suffixes)

        # Handle unmerged DataSet
        else:
            cplus  = [ c+suffixes[0] for c in columns ]
            cminus = [ c+suffixes[1] for c in columns ]
            dataset_plus.rename(columns=dict(zip(columns, cplus)), inplace=True)
            dataset_minus.rename(columns=dict(zip(columns, cminus)), inplace=True)
            result = dataset_plus.append(dataset_minus)
            # Fix dtypes -- NA values can cause upcast to object dtype
            result[cplus] = result[cplus].astype(dataset_plus[cplus].dtypes.to_dict())
            result[cminus] = result[cminus].astype(dataset_minus[cminus].dtypes.to_dict())

        if "M/ISYM" not in self.columns and self.merged:
            result.drop(columns="M/ISYM", inplace=True)
            
        return result.__finalize__(self)
        
    def hkl_to_asu(self, inplace=False):
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

        Returns
        -------
        DataSet

        See Also
        --------
        DataSet.hkl_to_observed : Opposite of DataSet.hkl_to_asu()
        """
        if inplace:
            dataset = self
        else:
            dataset = self.copy()

        # Compute new HKLs and phase shifts
        hkls = dataset.get_hkls()
        compressed_hkls, inverse = np.unique(hkls, axis=0, return_inverse=True)
        asu_hkls, isym, phi_coeff, phi_shift = hkl_to_asu(
            compressed_hkls, 
            dataset.spacegroup, 
            return_phase_shifts=True
        )
        index_keys = dataset.index.names
        dataset.reset_index(inplace=True)
        dataset[["H", "K", "L"]] = asu_hkls[inverse]
        dataset[["H", "K", "L"]] = dataset[["H", "K", "L"]].astype("HKL")
        dataset.set_index(index_keys, inplace=True)

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

        return dataset

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
        if inplace:
            dataset = self
        else:
            dataset = self.copy()

        # Validate input
        if m_isym is None:
            m_isym = dataset.get_m_isym_keys()
            if len(m_isym) == 1:
                m_isym = m_isym[0]
            else:
                raise ValueError(f"Method requires a single M/ISYM column -- found: {m_isym}")
        elif not isinstance(m_isym, str):
            raise ValueError("Provided M/ISYM column label should be type str")
        elif not isinstance(dataset.dtypes[m_isym], rs.M_IsymDtype):
            raise ValueError(f"Provided M/ISYM column label is of wrong dtype")

        # GH#3: Separate combined M/ISYM into M and ISYM        
        isym = (dataset[m_isym] % 256).to_numpy(dtype=np.int32)
        if not dataset.merged:
            dataset["PARTIAL"] = (dataset[m_isym]/256).astype(int) != 0
        dataset.drop(columns=m_isym, inplace=True)
        
        # Compute new HKLs and phase shifts
        hkls = dataset.get_hkls()
        hkls_isym = np.concatenate([hkls, isym.reshape(-1, 1)], axis=1)
        compressed, inverse = np.unique(hkls_isym, axis=0, return_inverse=True)
        observed_hkls, phi_coeff, phi_shift = hkl_to_observed(
            compressed[:, :3],       # compressed HKLs
            compressed[:, 3],        # compressed ISYM
            dataset.spacegroup,
            return_phase_shifts=True
        )
        index_keys = dataset.index.names
        dataset.reset_index(inplace=True)
        dataset[["H", "K", "L"]] = observed_hkls[inverse]
        dataset[["H", "K", "L"]] = dataset[["H", "K", "L"]].astype("HKL")
        dataset.set_index(index_keys, inplace=True)

        # Apply phase shift
        for k in dataset.get_phase_keys():
            dataset[k] = phi_coeff[inverse] * (dataset[k] + phi_shift[inverse])
        dataset.canonicalize_phases(inplace=True)
        
        return dataset
            
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
        if inplace:
            dataset = self
        else:
            dataset = self.copy()

        for k in dataset.get_phase_keys():
            dataset[k] = utils.canonicalize_phases(dataset[k])

        return dataset
