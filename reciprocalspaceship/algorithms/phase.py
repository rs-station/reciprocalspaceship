import numpy as np


def translate(H, phases, fractional_translation_vector, deg=True):
    """
    Compute new phases based on a real space translation vector
    in fractional coordinates. 

    Parameters
    ----------
    H : array
        nx3 array of miller indices. 
    phases : array
        lengh n array of phases.
    deg : bool (optional)
        phases are in degrees. 

    Returns
    -------
    translated_phases : array
        The phases updated based on the translation vector
    """ 
    phi = phases
    if deg:
        phi = np.deg2rad(phases)
    shift =  np.exp(2j * np.pi * H@fractional_translation_vector)
    translated_phases = np.angle(np.exp(1j*phi) * shift)
    if deg:
        translated_phases = np.rad2deg(translated_phases)
    return  translated_phases


def align_phases(H, phases, reference_phases, deg=True):
    """
    Apply one of standard set of translations to minimize the difference between
    two sets of phases. This corrects for a common pathology whereby degenerate
    origins can make it hard to compare phases.

    Parameters
    ----------
    H : array
        nx3 array of miller indices. 
    phases : array
        lengh n array of phases.
    deg : bool (optional)
        phases are in degrees. 

    Returns
    -------
    aligned_phases : array
    """
    v = np.array([0, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 5/6])
    v = np.stack(np.meshgrid(v,v,v)).reshape((3, -1))

    test_phases = translate(H, phases[...,None], v, deg)
    residuals = reference_phases[...,None] - test_phases
    idx = np.argmin(np.abs(residuals).sum(-2), -1)
    return translate(H, phases, v[:,idx], deg)

if __name__=="__main__":
    n = 100
    H = np.random.randint(0, 30, [n, 3])
    ref = 360. * (np.random.random(n) - 0.5)
    t = [1/4., 2/3., 5/6.]
    v = np.array([0, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 5/6])
    t = np.random.choice(v, 3)
    phases = translate(H, ref, t)
    test = align_phases(H, phases, ref)

    from IPython import embed
    embed(colors='Linux')

