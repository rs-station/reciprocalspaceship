import inspect
from pandas import DataFrame
import reciprocalspaceship as rs

def summarize_mtz_dtypes(print_summary=True):
    """
    Returns a table summarizing the MTZ Dtypes that can be used
    in a DataSet or DataSeries. These MTZ Dtypes are used to ensure 
    compatibility with the different column types supported by the 
    `MTZ file specification`_.

    .. _MTZ file specification: http://legacy.ccp4.ac.uk/html/mtzformat.html#coltypes

    Parameters
    ----------
    print_summary : bool
        Whether to print the summary table to stdout

    Returns
    -------
    pd.DataFrame
        If `print_summary` is False, returns DataFrame summarizing the MTZ 
        data types. Otherwise, returns None
    """
    dtypes = inspect.getmembers(rs.dtypes, inspect.isclass)
    data = []
    for dtype, hierarchy in dtypes:
        data.append((hierarchy.mtztype, hierarchy.name, dtype, hierarchy.type.__name__))
    df = DataFrame(data, columns=["MTZ Code", "Name", "Class", "Internal"])

    if print_summary:
        print(df.to_string(index=False))
        return
    else:
        return df
