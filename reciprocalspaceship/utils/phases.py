import numpy as np
import gemmi

def canonicalize_phases(phases, deg=True):
    """
    Place phases in the interval between -180 and 180. deg == True implies 
    degrees; False implies radians.
    """
    if deg == True:
        return (phases + 180.) % (2 * 180.) - 180.
    elif deg == False:
        return (phases + np.pi) % (2 * np.pi) - np.pi
    else:
        raise TypeError(f"deg has type {type(deg)}, but it should have type bool")

def get_phase_restrictions(H, spacegroup):
    """
    Return phase restrictions for Miller indices in a given space group.

    If there are no phase restrictions, an empty list is returned for that
    Miller index. If a given Miller index is systematically absent an
    empty list is also returned. 

    Parameters
    ----------
    H : array
        n x 3 array of Miller indices
    spacegroup : gemmi.SpaceGroup
        Space group for determining phase restrictions

    Returns
    -------
    restrictions : list of lists
         List of lists of phase restrictions for each Miller index. An empty
         list is returned for Miller indices without phase restrictions
    """
    from reciprocalspaceship.utils.asu import is_absent,is_centric
    from reciprocalspaceship.utils.symop import apply_to_hkl,phase_shift
    friedel_op = gemmi.Op("-x,-y,-z")
    #Grabs all the non-identity symops
    ops = spacegroup.operations().sym_ops[1:]
    restrictions = [[]] * len(H)
    #This is the case for P1
    if len(ops) == 0:
        return restrictions

    #Phase restrictions only apply to centrics. We'll also ignore any absent refls
    mask = (is_centric(H, spacegroup)) & (~is_absent(H, spacegroup))
    idx = np.where(mask)[0]
    h = H[mask,:]
    hits  = np.column_stack([np.all(apply_to_hkl(h, op) == apply_to_hkl(h, friedel_op), axis=1) for op in ops])

    hits[np.cumsum(hits, axis=-1) > 1] = False #Remove duplicate hits
    shifts = np.column_stack([np.rad2deg(phase_shift(h, op)) for op in ops])
    shifts = shifts[np.arange(len(hits)), hits.argmax(-1)]
    restriction = np.column_stack((
        shifts / 2.,
        180. + shifts / 2.
    ))
    restriction = canonicalize_phases(restriction)
    restriction.sort(-1)
    for i,_ in np.argwhere(hits):
        restrictions[idx[i]] = restriction[i].tolist()

    return restrictions

