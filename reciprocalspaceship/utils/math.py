import numpy as np

def angle_between(vec1, vec2, deg=True):
    """
    This function computes the angle between vectors along the last dimension of the input arrays.
    This version is a numerically stable one based on arctan2 as described in this post:
     - https://scicomp.stackexchange.com/a/27769/39858

    Parameters
    ----------
    vec1 : array
        An arbitrarily batched arry of vectors
    vec2 : array
        An arbitrarily batched arry of vectors
    deg : bool (optional)
        Whether angles are returned in degrees or radians. The default is degrees (deg=True).

    Returns
    -------
    angles : array
        A vector of angles with the same leading dimensions of vec1 and vec2.
    """
    v1 = vec1 / np.linalg.norm(vec1, axis=-1)[...,None]
    v2 = vec2 / np.linalg.norm(vec2, axis=-1)[...,None]
    x1 = np.linalg.norm(v1 - v2, axis=-1)
    x2 = np.linalg.norm(v1 + v2, axis=-1)
    alpha = 2.*np.arctan2(x1, x2)
    if deg:
        return np.rad2deg(alpha)
    return alpha


