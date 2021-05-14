import numpy as np
import pandas as pd
import reciprocalspaceship as rs

def compute_multiplicity(hobs, cell, spacegroup, anomalous=False, dmin=None):
    """
    Compute the multiplicity of all valid reflections in the reciprocal ASU.

    Parameters
    ----------
    hobs : np.array(int)
        An n by 3 array of observed miller indices which are not necessarily
        in the reciprocal asymmetric unit
    spacegroup : gemmi.SpaceGroup
        A gemmi SpaceGroup object.
    cell : gemmi.UnitCell
        A gemmi UnitCell object.
    anomalous : bool (optional)
        Whether or not the data are anomalous.
    dmin : float (optional)
        If no dmin is supplied, the maximum resolution reflection will be used.

    Returns
    -------
    multiplicity : rs.DataSet
    """
    hobs = hobs[~rs.utils.is_absent(hobs, spacegroup)]
    dhkl = rs.utils.compute_dHKL(hobs, cell)
    if dmin is None:
        dmin = dhkl.min()
    hobs = hobs[dhkl >= dmin]
    decimals = 5. #Round after this many decimals
    dmin = np.floor(dmin * 10**decimals) * 10**-decimals
    hobs,isym = rs.utils.hkl_to_asu(hobs, spacegroup)
    if anomalous:
        fminus = isym % 2 == 0
        hobs[fminus] = -hobs[fminus]

    mult = rs.DataSet({                     
	'H' : hobs[:,0],             
	'K' : hobs[:,1],             
	'L' : hobs[:,2],             
	'Count' : np.ones(len(hobs)),
    }, cell=cell, spacegroup=spacegroup).groupby(['H', 'K', 'L']).sum()

    hall = rs.utils.generate_reciprocal_asu(cell, spacegroup, dmin, anomalous)

    ASU = rs.DataSet({                     
	'H' : hall[:,0],             
	'K' : hall[:,1],             
	'L' : hall[:,2],             
	'Count' : np.zeros(len(hall)),
    }, cell=cell, spacegroup=spacegroup).groupby(['H', 'K', 'L']).sum()
    ASU = ASU.loc[ASU.index.difference(mult.index)]
    return mult.append(ASU)

