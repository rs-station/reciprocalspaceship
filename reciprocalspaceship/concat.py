import reciprocalspaceship as rs
import pandas as pd

def concat(*args, check_isomorphous=True, **kwargs):
    """
    Concatenate ``rs`` objects along a particular axis. This method 
    follows the behavior of ``pd.concat``. However, if DataSet objects are provided,
    attributes (such as `cell` and `spacegroup`) are set in the returned DataSet. 
    The returned attributes are inherited from the first object in `objs`.

    For the full documentation, see the `Pandas API Reference page`_. 

    .. _Pandas API Reference page: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

    Parameters
    ----------
    check_isomorphous : bool
        If `objs` to concatenate are instances of ``rs.DataSet``, their
        cell and spacegroup attributes will be compared to ensure they are
        isomorphous. 

    Returns
    -------
    rs.DataSet or rs.DataSeries
        Returns rs.DataSeries when concatenating rs.DataSeries along the
        index (axis=0). Returns rs.DataSet in all other cases.   

    See Also
    --------
    DataSet.append : Concatenate DataSets
    """
    objs = kwargs.get("objs", args[0])
    first = objs[0]
    if check_isomorphous and isinstance(first, rs.DataSet):
        for obj in objs[1:]:
            if not first.is_isomorphous(obj):
                raise ValueError("Provided DataSets are not isomorphous")
        
    result = pd.concat(*args, **kwargs)

    return result.__finalize__(first)
