import pandas as pd
import gemmi
from reciprocalspaceship import Crystal
from reciprocalspaceship.dtypes.mapping import mtzcode2dtype

def read(hklfile, a=None, b=None, c=None, alpha=None, beta=None,
         gamma=None, sg=None):
    """
    Initialize attributes and populate the crystal object with data from
    a HKL file of reflections. This is the output format used by 
    Precognition when processing Laue diffraction data.

    Parameters
    ----------
    hklfile : str or file
        name of an hkl file or a file object
    a : float
        edge length, a, of the unit cell
    b : float
        edge length, b, of the unit cell
    c : float
        edge length, c, of the unit cell
    alpha : float
        interaxial angle, alpha, of the unit cell
    beta : float
        interaxial angle, beta, of the unit cell
    gamma : float
        interaxial angle, gamma, of the unit cell
    sg : str or int
        If int, this should specify the space group number. If str, 
        this should be a space group symbol
    """
    # Read data from HKL file
    if hklfile.endswith(".hkl"):
        usecols = [0, 1, 2, 3, 4, 5, 6]
        F = pd.read_csv(hklfile, header=None, delim_whitespace=True,
                        names=["H", "K", "L", "F+", "SigF+", "F-", "SigF-"],
                        usecols=usecols)
        mtztypes = ["H", "H", "H", "G", "L", "G", "L"]
    elif hklfile.endswith("*.ii"):
        usecols = range(10)
        F = pd.read_csv(hklfile, header=None, delim_whitespace=True,
                        names=["H", "K", "L", "Multiplicity", "X", "Y",
                               "Resolution", "Wavelength", "I", "SigI"],
                        usecols=usecols)
        mtztypes = ["H", "H", "H", "I", "R", "R", "R", "R", "J", "Q"]

    # Check if any anomalous data is actually included
    if len(F["F-"].unique()) == 1:
        F = F[["H", "K", "L", "F+", "SigF+"]]
        F.rename(columns={"F+":"F", "SigF+":"SigF"}, inplace=True)
        mtztypes = ["H", "H", "H", "F", "Q"]
        
    crystal = Crystal()
    for (k,v), mtztype in zip(F.items(), mtztypes):
        crystal[k] = v
        crystal[k] = crystal[k].astype(mtzcode2dtype[mtztype])
    crystal.set_index(["H", "K", "L"], inplace=True)

    # Set Crystal attributes
    if (a and b and c and alpha and beta and gamma):
        crystal.cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)
    if sg:
        crystal.spacegroup = gemmi.SpaceGroup(sg)
        
    return crystal

def write(self, outfile, sf_key, err_key):
    """
    Write contents of Crystal object to an HKL file

    Parameters
    ----------
    outfile : str or file
        name of an hkl file or file-like object
    sf_key : str
        key for structure factor in DataFrame
    err_key : str
        key for structure factor error in DataFrame
    """

    with open(outfile, 'w') as out:
        for (h, k, l), d in self.iterrows():
            out.write(f"{h:5d}{k:5d}{l:5d}{d[sf_key]:15.3f}{d[err_key]:15.3f}\n")

    return
