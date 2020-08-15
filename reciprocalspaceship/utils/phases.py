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

    If there are no phase restrictions, an empty array is returned for that
    Miller index. If a given Miller index is systematically absent an
    empty array is also returned. 

    Parameters
    ----------
    H : array
        n x 3 array of Miller indices
    spacegroup : gemmi.SpaceGroup
        Space group for determining phase restrictions

    Returns
    -------
    restrictions : list of arrays
         list of arrays with phase restrictions for each Miller index
    """
    from reciprocalspaceship.utils.asu import hkl_is_absent
    from reciprocalspaceship.utils.structurefactors import is_centric
    
    restrictions = []
    for h in H:
        if not is_centric([h], spacegroup)[0] or hkl_is_absent([h], spacegroup)[0]:
            restrictions.append([])
        else:
            friedelop = gemmi.Op("-x,-y,-z")
            hit = False
            for op in spacegroup.operations().sym_ops[1:]:
                if op.apply_to_hkl(h) == friedelop.apply_to_hkl(h):
                    shift = np.rad2deg(op.phase_shift(h))
                    restriction = np.array([shift/2, 180+(shift/2)])
                    restriction = canonicalize_phases(restriction)
                    restriction.sort()
                    restrictions.append(restriction)
                    hit = True
                    break
                
            # Handle [0, 0, 0] in P1
            if not hit:
                restrictions.append([])
    return restrictions

