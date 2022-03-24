import warnings

import gemmi
import pandas as pd

from reciprocalspaceship import DataSet


def read_precognition(hklfile, spacegroup=None, cell=None, logfile=None):
    """
    Initialize attributes and populate the DataSet object with data from
    a precognition hkl or ii file of reflections. This is the output format used by
    Precognition when processing Laue diffraction data.

    Parameters
    ----------
    hklfile : str or file
        name of an hkl or ii file or a file object
    spacegroup : str or int
        If int, this should specify the space group number. If str,
        this should be a space group symbol
    cell : tuple or list of floats
        Unit cell parameters
    logfile : str or file
        name of a log file to parse to get cell parameters and spacegroup. Only
        used when spacegroup and/or cell are not explicitly provided.
    """
    # Read data from HKL file
    if hklfile.endswith(".hkl"):
        usecols = [0, 1, 2, 3, 4, 5, 6]
        F = pd.read_csv(
            hklfile,
            header=None,
            delim_whitespace=True,
            names=["H", "K", "L", "F(+)", "SigF(+)", "F(-)", "SigF(-)"],
            usecols=usecols,
        )
        mtztypes = ["H", "H", "H", "G", "L", "G", "L"]

        # Check if any anomalous data is actually included
        if len(F["F(-)"].unique()) == 1:
            F = F[["H", "K", "L", "F(+)", "SigF(+)"]]
            F.rename(columns={"F(+)": "F", "SigF(+)": "SigF"}, inplace=True)
            mtztypes = ["H", "H", "H", "F", "Q"]

    # Read data from II file
    elif hklfile.endswith(".ii"):
        usecols = range(10)
        F = pd.read_csv(
            hklfile,
            header=None,
            delim_whitespace=True,
            names=[
                "H",
                "K",
                "L",
                "Multiplicity",
                "X",
                "Y",
                "Resolution",
                "Wavelength",
                "I",
                "SigI",
            ],
            usecols=usecols,
        )
        mtztypes = ["H", "H", "H", "I", "R", "R", "R", "R", "J", "Q"]

    # Limit use to supported file formats
    else:
        raise ValueError("rs.read_precognition() only supports .ii and .hkl files")

    # If logfile is given, read cell parameters and spacegroup
    # Assign these as temporary variables, and determine priority later.

    if logfile:
        from os.path import basename

        with open(logfile, "r") as log:
            lines = log.readlines()

        # Read spacegroup
        sgline = [l for l in lines if "space-group" in l][0]
        spacegroup_from_log = [s for s in sgline.split() if "#" in s][0].lstrip("#")

        # Read cell parameters
        block = [i for i, l in enumerate(lines) if basename(hklfile) in l][0]
        lengths = lines[block - 19].split()[-3:]
        a, b, c = map(float, lengths)
        angles = lines[block - 18].split()[-3:]
        alpha, beta, gamma = map(float, angles)
        cell_from_log = (a, b, c, alpha, beta, gamma)

    dataset = DataSet(F)
    dataset = dataset.astype(dict(zip(dataset.columns, mtztypes)))
    dataset.set_index(["H", "K", "L"], inplace=True)

    # Set DataSet attributes
    # Prioritize explicitly supplied arguments
    if cell:
        dataset.cell = cell
    elif logfile:
        dataset.cell = cell_from_log

    if spacegroup:
        dataset.spacegroup = spacegroup
    elif logfile:
        dataset.spacegroup = spacegroup_from_log

    if cell and spacegroup and logfile:
        warnings.warn("Ignoring logfile, as cell and spacegroup are both provided")

    return dataset
