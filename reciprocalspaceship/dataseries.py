import pandas as pd
import reciprocalspaceship as rs

class DataSeries(pd.Series):
    """
    One-dimensional ndarray with axis labels, representing a slice
    of a DataSet. DataSeries objects inherit methods from ``pd.Series``,
    and as such have support for statistical methods that automatically
    exclude missing data (represented as NaN).

    Operations between DataSeries align values on their associated index
    values, and as such do not need to have the same length. 

    For more information on the attributes and methods available with
    DataSeries objects, see the `Pandas documentation 
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html>`_.
    """
    
    @property
    def _constructor(self):
        return DataSeries

    @property
    def _constructor_expanddim(self):
        return rs.DataSet

    def to_friedel_dtype(self):
        """
        Return DataSeries with dtype set to Friedel equivalent. For example, 
        rs.IntensityDtype is converted to rs.FriedelIntensityDtype(). If
        there is not a Friedel equivalent dtype, the DataSeries is returned
        unchanged.
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
        Return DataSeries with dtype set from Friedel equivalent. For example, 
        rs.FriedelIntensityDtype is converted to rs.IntensityDtype(). If
        there is not a Friedel equivalent dtype, the DataSeries is returned 
        unchanged.
        """
        if isinstance(self.dtype, rs.FriedelStructureFactorAmplitudeDtype):
            return self.astype(rs.StructureFactorAmplitudeDtype())
        elif isinstance(self.dtype, rs.FriedelIntensityDtype):
            return self.astype(rs.IntensityDtype())
        elif (isinstance(self.dtype, rs.StandardDeviationFriedelIDtype) or
              isinstance(self.dtype, rs.StandardDeviationFriedelSFDtype)):
            return self.astype(rs.StandardDeviationDtype())
        return self
