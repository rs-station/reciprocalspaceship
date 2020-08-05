import numpy as np
import pandas as pd
import gemmi
from reciprocalspaceship import DataSet
from reciprocalspaceship.dtypes.base import MTZDtype

def from_gemmi(gemmi_mtz):
    """
    Construct DataSet from gemmi.Mtz object
    
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
    gemmi_mtz : gemmi.Mtz
        gemmi Mtz object

    Returns
    -------
    rs.DataSet
    """
    dataset = DataSet(spacegroup=gemmi_mtz.spacegroup, cell=gemmi_mtz.cell)

    # Build up DataSet
    for c in gemmi_mtz.columns:
        dataset[c.label] = c.array
        # Special case for CENTRIC and PARTIAL flags
        if c.type == "I" and c.label in ["CENTRIC", "PARTIAL"]:
            dataset[c.label] = dataset[c.label].astype(bool)
        else:
            dataset[c.label] = dataset[c.label].astype(c.type)
    dataset.set_index(["H", "K", "L"], inplace=True)

    # Handle unmerged DataSet. Raise ValueError if M/ISYM column is not unique
    m_isym = dataset.get_m_isym_keys()
    if m_isym and dataset.index.duplicated().any():
        if len(m_isym) == 1:
            dataset.merged = False
            dataset.hkl_to_observed(m_isym[0], inplace=True)
        else:
            raise ValueError("Only a single M/ISYM column is supported for unmerged data")
    else:
        dataset.merged = True
        
    return dataset

def to_gemmi(dataset, skip_problem_mtztypes=False):
    """
    Construct gemmi.Mtz object from DataSet

    If ``dataset.merged == False``, the reflections will be mapped to the
    reciprocal space ASU, and a M/ISYM column will be constructed. 

    If boolean flags with the label ``PARTIAL`` or ``CENTRIC`` are found
    in the DataSet, these will be cast to the ``MTZInt`` dtype, and included
    in the gemmi.Mtz object. 

    Parameters
    ----------
    dataset : rs.DataSet
        DataSet object to convert to gemmi.Mtz
    skip_problem_mtztypes : bool
        Whether to skip columns in DataSet that do not have specified
        mtz datatypes

    Returns
    -------
    gemmi.Mtz
    """
    # Check that cell and spacegroup are defined
    if not dataset.cell:
        raise AttributeError(f"Instance of type {dataset.__class__.__name__} has no unit cell information")
    if not dataset.spacegroup:
        raise AttributeError(f"Instance of type {dataset.__class__.__name__} has no space group information")

    # Build up a gemmi.Mtz object
    mtz = gemmi.Mtz()
    mtz.cell = dataset.cell
    mtz.spacegroup = dataset.spacegroup

    # Handle Unmerged data
    if not dataset.merged:
        dataset.hkl_to_asu(inplace=True)
    
    # Construct data for Mtz object. 
    mtz.add_dataset("reciprocalspaceship")
    temp = dataset.reset_index()
    columns = []
    for c in temp.columns:
        cseries = temp[c]
        if isinstance(cseries.dtype, MTZDtype):
            mtzcol = mtz.add_column(label=c, type=cseries.dtype.mtztype)
            columns.append(c)
        # Special case for CENTRIC and PARTIAL flags
        elif cseries.dtype.name == "bool" and c in ["CENTRIC", "PARTIAL"]:
            temp[c] = temp[c].astype("MTZInt")
            mtzcol = mtz.add_column(label=c, type="I")
            columns.append(c)
        elif skip_problem_mtztypes:
            continue
        else:
            raise ValueError(f"column of type {cseries.dtype} cannot be written to an MTZ file. "
                             f"To skip columns without explicit MTZ dtypes, set skip_problem_mtztypes=True")
    mtz.set_data(temp[columns].to_numpy(dtype="float32"))

    return mtz
    
def read_mtz(mtzfile):
    """
    Populate the dataset object with data from an MTZ reflection file.

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
    mtzfile : str or file
        name of an mtz file or a file object

    Returns
    -------
    DataSet
    """
    gemmi_mtz = gemmi.read_mtz_file(mtzfile)
    return from_gemmi(gemmi_mtz)

def write_mtz(dataset, mtzfile, skip_problem_mtztypes=False):
    """
    Write an MTZ reflection file from the reflection data in a DataSet.

    If ``dataset.merged == False``, the reflections will be mapped to the
    reciprocal space ASU, and a M/ISYM column will be constructed. 

    If boolean flags with the label ``PARTIAL`` or ``CENTRIC`` are found
    in dataset, these will be cast to the ``MTZInt`` dtype, and included
    in the output MTZ file. 

    Parameters
    ----------
    dataset : DataSet
        DataSet object to be written to MTZ file
    mtzfile : str or file
        name of an mtz file or a file object
    skip_problem_mtztypes : bool
        Whether to skip columns in DataSet that do not have specified
        MTZ datatypes
    """
    mtz = to_gemmi(dataset, skip_problem_mtztypes)
    mtz.write_to_file(mtzfile)
    return
