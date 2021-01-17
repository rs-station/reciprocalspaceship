import pandas as pd
from reciprocalspaceship import DataSet

def read_csv(csvfile, spacegroup=None, cell=None, merged=None,
             infer_mtz_dtypes=True, *args, **kwargs):
    """
    Read a comma-separated values (csv) file into a DataSet.

    This method also supports the arguments to `pandas.read_csv <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html>`_. See their documentation
    for more details.

    Parameters
    ----------
    csvfile : str, path object, or file-like object
        File to read as CSV file
    spacegroup : gemmi.SpaceGroup, int, or str
        Spacegroup with which to initialize DataSet
    cell : gemmi.UnitCell, tuple, list, or array-like
        Cell parameters with which to initialize DataSet
    merged : bool
        Whether DataSet contains merged reflection data
    infer_mtz_dtypes : bool
        Whether to infer MTZ dtypes based on column names

    Returns
    -------
    rs.DataSet
    """
    df = pd.read_csv(csvfile, *args, **kwargs)
    ds = DataSet(df, spacegroup=spacegroup, cell=cell, merged=merged)
    if infer_mtz_dtypes:
        ds.infer_mtz_dtypes(inplace=True)
    return ds
