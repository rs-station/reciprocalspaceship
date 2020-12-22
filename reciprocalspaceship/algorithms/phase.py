import numpy as np
import gemmi


#All credit to apeck12 for the following helpful function!
def find_equivalent_origins(spacegroup, sampling_interval=0.1):
    """
    Determine fractional coordinates corresponding to equivalent phase
    origins for the space group of interest.
    Parameters
    ----------
    spacegroup : gemmi.SpaceGroup
        gemmi.SpaceGroup instance
    sampling_interval : float (optional)
        search interval for polar axes as fractional cell length
    Returns
    -------
    eq_origins : array
        nx3 array of equivalent fractional origins
    """

    permitted = dict()
    for axis in ['x','y','z']:
        permitted[axis] = []

    # add shift for each symmetry element that contains an inversion
    sym_ops = spacegroup.operations()
    ops = [op.triplet() for op in sym_ops]
    for op in ops:
        if "-" in op:
            for element in op.split(","):
                if "-" in element:
                    if any(i.isdigit() for i in element):
                        axis, frac = element.split("+")
                        permitted[axis[1]].append(float(frac[0]) / float(frac[2]))
                    else:
                        permitted[element.split("-")[1][0]].append(0)

    # eliminate redundant entries and expand polar axes
    for axis in permitted.keys():
        permitted[axis] = np.unique(permitted[axis])
        if len(permitted[axis]) == 0:
            permitted[axis] = np.arange(0,1,sampling_interval)

    # get all combinations from each axis
    eq_origins = np.stack(np.meshgrid(permitted['x'],
                                      permitted['y'],
                                      permitted['z'])).reshape((3, -1)).T

    # add centering operations
    cen_ops = np.array(list(sym_ops.cen_ops)) / 24.0
    eq_origins = eq_origins + cen_ops[:,np.newaxis]
    eq_origins = eq_origins.reshape(-1, eq_origins.shape[-1])
    eq_origins = np.delete(eq_origins, np.where(eq_origins==1)[0], axis=0)

    return np.unique(eq_origins, axis=0)


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


def align_phases(H, phases, reference_phases, spacegroup, deg=True, sampling_interval=0.001):
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
    spacegroup : gemmi.SpaceGroup
        A gemmi space group object to help define the possible translation vectors.
    deg : bool (optional)
        phases are in degrees. 
    sampling_interval : float
        Fractional cell distance over which to sample. This is relevant only for the Polar space groups.

    Returns
    -------
    aligned_phases : array
    """
    v = find_equivalent_origins(spacegroup, sampling_interval=sampling_interval).T
    print(v)

    def loss_func(tvec):
        test_phases = translate(H, phases[...,None], tvec, deg)
        if deg:
            residuals = np.exp(1j * np.deg2rad(test_phases)) - np.exp(1j * np.deg2rad(reference_phases[...,None]))
        else:
            residuals = np.exp(1j * test_phases) - np.exp(1j * reference_phases[...,None])
        residuals = np.square(np.real(residuals)) + np.square(np.imag(residuals))
        return residuals.sum(-2)

    loss_vals = loss_func(v)
    idx = np.argmin(loss_vals, -1)
    solution = v[:,idx]
    return translate(H, phases, solution, deg)

if __name__=="__main__":
    n = 1000
    H = np.random.randint(0, 30, [n, 3])
    ref = 360. * (np.random.random(n) - 0.5)
    spacegroup = gemmi.SpaceGroup(173)
    t = np.array([0., 0., 1/13])
    phases = translate(H, ref, t)
    test = align_phases(H, phases, ref, spacegroup)
    from matplotlib import pyplot as plt
    plt.plot(ref, phases, 'k.', label='Shifted')
    plt.plot(ref, test, 'r.', label='Aligned')
    plt.legend()
    plt.show()

    from IPython import embed
    embed(colors='Linux')

