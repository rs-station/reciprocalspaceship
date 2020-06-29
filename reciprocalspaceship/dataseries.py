import pandas as pd
import reciprocalspaceship as rs

class DataSeries(pd.Series):
    """
    One-dimensional ndarray with axis labels, representing a slice
    of a DataSet. DataSeries objects inherit methods from ``pd.Series``,
    and as such have support for statistical methods that automatically
    exclude missing data (represented as NaN).

    Operations between DataSeries align values on their associated index
    values so they do not need to have the same length. 

    For more information on the attributes and methods available with
    DataSeries objects, see the `Pandas documentation 
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html>`_.

    Parameters
    ----------
    data : array-like, Iterable, dict, or scalar value
        data to be stored in DataSeries.
    index : array-like or Index
        Values must be hashable and have the same length as `data`.
        Non-unique index values are allowed. Will default to
        RangeIndex (0, 1, 2, ..., n) if not provided. If a dict is provided 
        as `data` and a `index` is given, `index` will override the keys
        found in the dict.
    dtype : str, numpy.dtype, or ExtensionDtype, optional
        Data type for the DataSeries.
    name : str, optional
        The name to give to the DataSeries.
    copy : bool, default False
        Copy input data.
    """
    
    @property
    def _constructor(self):
        return DataSeries

    @property
    def _constructor_expanddim(self):
        return rs.DataSet

    def to_friedel_dtype(self):
        """
        Convert dtype of DataSeries to the Friedel equivalent. If there
        is not a Friedel equivalent dtype, the DataSeries is returned 
        unchanged.

        Returns
        -------
        DataSeries

        See Also
        --------
        DataSeries.from_friedel_dtype : Convert dtype of DataSeries from the Friedel equivalent.

        Examples
        --------
        DataSeries has a Friedel equivalent:

        >>> s = rs.DataSeries([1, 2, 3], dtype="Intensity")
        >>> s
        0   1.0
        1   2.0
        2   3.0
        dtype: Intensity
        >>> s.to_friedel_dtype()
        0   1.0
        1   2.0
        2   3.0
        dtype: FriedelIntensity

        DataSeries does not have a Friedel equivalent:

        >>> s = rs.DataSeries([1, 2, 3], dtype="HKL")
        >>> s
        0    1
        1    2
        2    3
        dtype: HKL
        >>> s.to_friedel_dtype()
        0    1
        1    2
        2    3
        dtype: HKL
        """
        if isinstance(self.dtype, rs.StructureFactorAmplitudeDtype):
            return self.astype(rs.FriedelStructureFactorAmplitudeDtype())
        elif isinstance(self.dtype, rs.IntensityDtype):
            return self.astype(rs.FriedelIntensityDtype())
        elif (isinstance(self.dtype, rs.StandardDeviationDtype) and
              "SIGF" in self.name.upper()):
            return self.astype(rs.StandardDeviationFriedelSFDtype())
        elif (isinstance(self.dtype, rs.StandardDeviationDtype) and
              "SIGI" in self.name.upper()):
            return self.astype(rs.StandardDeviationFriedelIDtype())
        return self

    def from_friedel_dtype(self):
        """
        Convert dtype of DataSeries from the Friedel equivalent. If
        DataSeries is not a Friedel-related dtype, it is returned 
        unchanged.

        Returns
        -------
        DataSeries

        See Also
        --------
        DataSeries.to_friedel_dtype : Convert dtype of DataSeries to the Friedel equivalent

        Examples
        --------
        DataSeries has a Friedel equivalent:

        >>> s = rs.DataSeries([1, 2, 3], dtype="FriedelIntensity")
        >>> s
        0   1.0
        1   2.0
        2   3.0
        dtype: FriedelIntensity
        >>> s.from_friedel_dtype()
        0   1.0
        1   2.0
        2   3.0
        dtype: Intensity

        DataSeries does not have a Friedel equivalent:

        >>> s = rs.DataSeries([1, 2, 3], dtype="HKL")
        >>> s
        0    1
        1    2
        2    3
        dtype: HKL
        >>> s.from_friedel_dtype()
        0    1
        1    2
        2    3
        dtype: HKL

        """
        if isinstance(self.dtype, rs.FriedelStructureFactorAmplitudeDtype):
            return self.astype(rs.StructureFactorAmplitudeDtype())
        elif isinstance(self.dtype, rs.FriedelIntensityDtype):
            return self.astype(rs.IntensityDtype())
        elif (isinstance(self.dtype, rs.StandardDeviationFriedelIDtype) or
              isinstance(self.dtype, rs.StandardDeviationFriedelSFDtype)):
            return self.astype(rs.StandardDeviationDtype())
        return self

    def infer_mtz_dtype(self):
        """
        Infer MTZ dtype from column name and underlying data.
        
        If name does not match a common MTZ column, the method will return 
        an MTZInt or MTZReal depending on whether the data is composed of 
        integers or floats, respectively. If the data is non-numeric, 
        the returned dtype will be unchanged. If input dataseries is 
        already a MTZDtype, it will be returned unchanged.
        
        Returns
        -------
        DataSeries

        See Also
        --------
        DataSet.infer_mtz_dtypes : Infer MTZ dtypes for columns in DataSet

        Examples
        --------
        Common intensity column name:

        >>> s = rs.DataSeries([1, 2, 3], name="I")
        >>> s.infer_mtz_dtype()
        0   1.0
        1   2.0
        2   3.0
        Name: I, dtype: Intensity

        Common intensity column name for anomalous data:

        >>> s = rs.DataSeries([1, 2, 3], name="I(+)")
        >>> s.infer_mtz_dtype()
        0   1.0
        1   2.0
        2   3.0
        Name: I(+), dtype: FriedelIntensity
        
        Ambiguous name case:

        >>> s = rs.DataSeries([1, 2, 3], name="Something")
        >>> s.infer_mtz_dtype()
        0    1
        1    2
        2    3
        Name: Something, dtype: MTZInt
        """
        from reciprocalspaceship.dtypes.inference import infer_mtz_dtype
        return infer_mtz_dtype(self)
