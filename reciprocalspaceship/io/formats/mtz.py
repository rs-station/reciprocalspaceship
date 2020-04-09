import numpy as np
import pandas as pd
import gemmi
from reciprocalspaceship import DataSet
from reciprocalspaceship.dtypes.mapping import mtzcode2dtype

def read(mtzfile):
    """
    Populate the dataset object with data from an MTZ reflection file.

    Parameters
    ----------
    mtzfile : str or file
        name of an mtz file or a file object
    """
    mtzgemmi = gemmi.read_mtz_file(mtzfile)

    dataset = DataSet()
    
    for c in mtzgemmi.columns:
        dataset[c.label] = c.array
        dataset[c.label] = dataset[c.label].astype(mtzcode2dtype[c.type])
    dataset.set_index(["H", "K", "L"], inplace=True)
    
    # Set DataSet attributes
    dataset.cell = mtzgemmi.cell
    dataset.spacegroup = mtzgemmi.spacegroup

    return dataset

def write(dataset, mtzfile, skip_problem_mtztypes=False):
    """
    Write an MTZ reflection file from the reflection data in a DataSet.

    Parameters
    ----------
    mtzfile : str or file
        name of an mtz file or a file object
    skip_problem_mtztypes : bool
        Whether to skip columns in DataSet that do not have specified
        mtz datatypes
    """
    # Check that cell and spacegroup are defined
    if not dataset.cell:
        raise AttributeError(f"Instance of type {dataset.__class__.__name__} has no unit cell information")
    if not dataset.spacegroup:
        raise AttributeError(f"Instance of type {dataset.__class__.__name__} has no space group information")

    # Build up a Gemmi MTZ object
    mtz = gemmi.Mtz()
    mtz.cell = dataset.cell
    mtz.spacegroup = dataset.spacegroup
    
    mtz.add_dataset("reciprocalspaceship")
    temp = dataset.reset_index()
    columns = []
    for c in temp.columns:
        try:
            cseries = temp[c]
            mtztype = cseries.dtype.mtztype
            mtzcol = mtz.add_column(label=c, type=mtztype)
            columns.append(c)
        except AttributeError:
            if skip_problem_mtztypes:
                continue
            else:
                raise AttributeError(f"'numpy.dtype' object has no attribute 'mtztype'\n\n"
                                     f"To skip columns without explicit mtztypes, set skip_problem_mtztypes=True")
            
    mtz.set_data(temp[columns].to_numpy(dtype="float32"))

    # Write MTZ
    mtz.write_to_file(mtzfile)
    
    return
