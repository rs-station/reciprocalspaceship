import numpy as np
import pandas as pd
import gemmi

def read(self, mtzfile):
    """
    Populate the crystal object with data from an MTZ reflection file.

    Parameters
    ----------
    mtzfile : str or file
        name of an mtz file or a file object
    """
    mtzgemmi = gemmi.read_mtz_file(mtzfile)

    # Copy data
    all_data = np.array(mtzgemmi, copy=False)
    F = pd.DataFrame(data=all_data, columns=mtzgemmi.column_labels())

    for k,v in F.items():
        self[k] = v
    self.set_index(["H", "K", "L"], inplace=True)

    
    # Set Crystal attributes
    self.cell = mtzgemmi.cell
    self.spacegroup = mtzgemmi.spacegroup

    return self
