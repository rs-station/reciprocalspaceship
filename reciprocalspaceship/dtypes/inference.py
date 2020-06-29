from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from reciprocalspaceship.dtypes.base import MTZDtype

def infer_mtz_dtype(dataseries):
    """
    Infer MTZ dtype from column name and underlying data.

    If name does not match a common MTZ column, the method will return 
    an MTZInt or MTZReal depending on whether the data is composed of 
    integers or floats, respectively. If the data is non-numeric, the 
    returned dtype will be unchanged.

    Notes
    -----
    - If input dataseries is already a MTZDtype, it will be returned
      unchanged

    Parameters
    ----------
    dataseries : DataSeries
        Input DataSeries for which to infer dtype

    Returns
    -------
    DataSeries
        DataSeries with the inferred dtype
    """
    name = dataseries.name
    dtype = dataseries.dtype
    
    # If dtype is object, try to coerce to more informative dtype
    if is_object_dtype(dataseries.dtype):
        new = dataseries.convert_dtypes()
        dtype = new.dtype
    
    if isinstance(dataseries.dtype, MTZDtype):
        return dataseries

    # Name is None
    elif name is None:
        if is_integer_dtype(dtype):
            return dataseries.astype("I")
        elif is_float_dtype(dtype):
            return dataseries.astype("R")
        else:
            return dataseries

    elif (name.upper() == "H" or name.upper() == "K" or name.upper() == "L"):
        return dataseries.astype("H")

    elif name.upper().startswith("HL"):
        return dataseries.astype("A")

    elif name.upper().startswith("PH"):
        return dataseries.astype("P")

    elif name.upper() == "E":
        return dataseries.astype("E")

    elif "BATCH" in name.upper() or "IMAGE" in name.upper():
        return dataseries.astype("B")

    elif "M/ISYM" in name.upper():
        return dataseries.astype("Y")
    
    elif name.upper().startswith("WEIGHT") or name.upper() == "W":
        return dataseries.astype("W")
    
    elif name.upper().startswith("SIG"):
        # Check Friedel
        if any(match in name for match in ["(+)", "(-)"]):
            # Check SF or I
            if "F" in name.upper():
                return dataseries.astype("L")
            else:
                return dataseries.astype("M")
        else:
            return dataseries.astype("Q")

    elif name.upper().startswith("I"):
        if any(match in name for match in ["(+)", "(-)"]):
            return dataseries.astype("K")
        else:
            return dataseries.astype("J")

    elif name.upper().startswith("FREE"):
        return dataseries.astype("I")
        
    elif (name.upper().startswith("F") or "ANOM" in name.upper()):
        if any(match in name for match in ["(+)", "(-)"]):
            return dataseries.astype("G")
        else:
            return dataseries.astype("F")
        
    # Fall back to general dtypes
    if is_integer_dtype(dtype):
        return dataseries.astype("I")
    elif is_float_dtype(dtype):
        return dataseries.astype("R")
    else:
        return dataseries

        
