import numpy as np
import pandas as pd
import gemmi
from reciprocalspaceship import Crystal

def read(mtzfile):
    """
    Populate the crystal object with data from an MTZ reflection file.

    Parameters
    ----------
    mtzfile : str or file
        name of an mtz file or a file object
    """
    mtzgemmi = gemmi.read_mtz_file(mtzfile)

    # # Copy data
    # all_data = np.array(mtzgemmi, copy=False)
    # F = pd.DataFrame(data=all_data, columns=mtzgemmi.column_labels())

    crystal = Crystal()
    
    for c in mtzgemmi.columns:
        crystal[c.label] = c.array
        crystal[c.label].mtztype = c.type
        print(crystal[c.label].mtztype)
    crystal.set_index(["H", "K", "L"], inplace=True)
    print(crystal[c.label].mtztype)
    
    # Set Crystal attributes
    crystal.cell = mtzgemmi.cell
    crystal.spacegroup = mtzgemmi.spacegroup

    return crystal
