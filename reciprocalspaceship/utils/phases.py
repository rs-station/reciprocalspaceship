from numpy import pi

def canonicalize_phases(phases, deg=True):
    """
    Place phases in the interval between -180 and 180. deg == True implies degrees; False implies radians.
    """
    if deg == True:
        return (phases + 180.) % (2 * 180.) - 180.
    elif deg == False:
        return (phases + pi) % (2 * pi) - pi
    else:
        raise TypeError(f"deg has type {type(deg)}, but it should have type bool")
