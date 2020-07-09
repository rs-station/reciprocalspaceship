import numpy as np
import pandas as pd
import gemmi
from reciprocalspaceship import DataSet
from reciprocalspaceship.dtypes.base import MTZDtype

def from_gemmi(gemmi_mtz):
    """
    Construct DataSet from gemmi.Mtz object

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
        dataset[c.label] = dataset[c.label].astype(c.type)
    dataset.set_index(["H", "K", "L"], inplace=True)

    # Handle unmerged DataSet. It is assumed that there is a single
    # M/ISYM column
    if "M/ISYM" in dataset.dtypes:
        dataset.merged = False
        m_isym = dataset.dtypes[dataset.dtypes == "M/ISYM"].index.values[0]
        dataset.hkl_to_observed(m_isym, inplace=True)
    else:
        dataset.merged = True
        
    return dataset

def to_gemmi(dataset, skip_problem_mtztypes=False):
    """
    Construct gemmi.Mtz object from DataSet

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

    Parameters
    ----------
    mtzfile : str or file
        name of an mtz file or a file object
    """
    gemmi_mtz = gemmi.read_mtz_file(mtzfile)
    return from_gemmi(gemmi_mtz)

def write_mtz(dataset, mtzfile, skip_problem_mtztypes=False):
    """
    Write an MTZ reflection file from the reflection data in a DataSet.

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
