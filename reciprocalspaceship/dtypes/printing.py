import inspect
from pandas import DataFrame
import reciprocalspaceship as rs

def print_mtz_dtypes():
    """
    Prints a table summarizing the MTZ Dtypes that can be used
    in a DataSet or DataSeries. These MTZ Dtypes are used to ensure 
    compatibility with the different column types supported by the 
    `MTZ file specification`_.

    .. _MTZ file specification: http://www.ccp4.ac.uk/html/mtzformat.html#coltypes
    """
    dtypes = inspect.getmembers(rs.dtypes, inspect.isclass)
    data = []
    for dtype, hierarchy in dtypes:
        data.append((hierarchy.mtztype, hierarchy.name, dtype))
    df = DataFrame(data, columns=["MTZ Code", "Name", "Class"])
    print(df.to_string(index=False))
    return
