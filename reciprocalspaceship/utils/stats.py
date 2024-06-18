import numpy as np

import reciprocalspaceship as rs
from reciprocalspaceship.decorators import cellify, spacegroupify


@cellify
@spacegroupify
def compute_redundancy(
    hobs, cell, spacegroup, full_asu=True, anomalous=False, dmin=None
):
    """
    Compute the multiplicity of all valid reflections in the reciprocal ASU.

    Parameters
    ----------
    hobs : np.array(int)
        An n by 3 array of observed miller indices which are not necessarily
        in the reciprocal asymmetric unit
    cell : tuple, list, np.ndarray of cell parameters, or gemmi.UnitCell
        Unit cell parameters
    spacegroup : str, int, or gemmi.SpaceGroup
        The space group of the dataset
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
    hobs, isym = rs.utils.hkl_to_asu(hobs, spacegroup)
    if anomalous:
        fminus = isym % 2 == 0
        hobs[fminus] = -hobs[fminus]

    mult = (
        rs.DataSet(
            {
                "H": hobs[:, 0],
                "K": hobs[:, 1],
                "L": hobs[:, 2],
                "Count": np.ones(len(hobs)),
            },
            cell=cell,
            spacegroup=spacegroup,
        )
        .groupby(["H", "K", "L"])
        .sum()
    )

    if full_asu:
        hall = rs.utils.generate_reciprocal_asu(cell, spacegroup, dmin, anomalous)

        ASU = rs.DataSet(
            {
                "H": hall[:, 0],
                "K": hall[:, 1],
                "L": hall[:, 2],
                "Count": np.zeros(len(hall)),
            },
            cell=cell,
            spacegroup=spacegroup,
        ).set_index(["H", "K", "L"])
        ASU = ASU.loc[ASU.index.difference(mult.index)]
        mult = rs.concat([mult, ASU])

    mult = mult.sort_index()
    return mult.get_hkls(), mult["Count"].to_numpy(np.int32)


def weighted_pearsonr(x, y, w):
    """
    Calculate a [weighted Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient).

    Note
    ----
    x, y, and w may have arbitrarily shaped leading dimensions. The correlation coefficient will always be computed pairwise along the last axis.

    Parameters
    ----------
    x : np.array(float)
        An array of observations.
    y : np.array(float)
        An array of observations the same shape as x.
    w : np.array(float)
        An array of weights the same shape as x. These needn't be normalized.

    Returns
    -------
    r : float
        The Pearson correlation coefficient along the last dimension. This has shape {x,y,w}.shape[:-1].
    """
    z = np.reciprocal(w.sum(-1))

    mx = z * np.einsum("...a,...a->...", w, x)
    my = z * np.einsum("...a,...a->...", w, y)

    dx = x - np.expand_dims(mx, axis=-1)
    dy = y - np.expand_dims(my, axis=-1)

    cxy = z * np.einsum("...a,...a,...a->...", w, dx, dy)
    cx = z * np.einsum("...a,...a,...a->...", w, dx, dx)
    cy = z * np.einsum("...a,...a,...a->...", w, dy, dy)

    r = cxy / np.sqrt(cx * cy)
    return r
