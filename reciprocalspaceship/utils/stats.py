import numpy as np
import pandas as pd
import reciprocalspaceship as rs

def compute_redundancy(hobs, cell, spacegroup, full_asu=True, anomalous=False, dmin=None):
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
    full_asu : bool (optional)
        Include all the reflections in the calculation irrespective of whether they were
        observed.
    anomalous : bool (optional)
        Whether or not the data are anomalous.
    dmin : float (optional)
        If no dmin is supplied, the maximum resolution reflection will be used.

    Returns
    -------
    hunique : np.array (int32)
        An n by 3 array of unique miller indices from hobs.
    counts : np.array (int32)
        A length n array of counts for each miller index in hunique.
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

    if full_asu:
        hall = rs.utils.generate_reciprocal_asu(cell, spacegroup, dmin, anomalous)

        ASU = rs.DataSet({                     
            'H' : hall[:,0],             
            'K' : hall[:,1],             
            'L' : hall[:,2],             
            'Count' : np.zeros(len(hall)),
        }, cell=cell, spacegroup=spacegroup).set_index(['H', 'K', 'L'])
        ASU = ASU.loc[ASU.index.difference(mult.index)]
        mult = mult.append(ASU)

    mult = mult.sort_index()
    return mult.get_hkls(), mult['Count'].to_numpy(np.int32)

