import numpy as np

def merge(dataset, intensity_key="I", sigma_key="SIGI", sort=False):
    """
    Merge dataset using inverse-variance weights.

    Parameters
    ----------
    dataset : rs.DataSet
        Unmerged DataSet containing scaled intensities and uncertainties
    intensity_key : str
        Column name for intensities
    sigma_key : str
        Column name for uncertainties
    sort : bool
        If True, index of returned DataSet will be sorted

    Returns
    -------
    rs.DataSet
        Merged DataSet object
    """

    if dataset.merged:
        raise ValueError("rs.algorithms.merge() can only be called with an unmerged DataSet")

    # Map observations to reciprocal ASU
    ds = dataset.hkl_to_asu(anomalous=True)
    ds["w"]  = ds[sigma_key]**-2
    ds["wI"] = ds[intensity_key] * ds["w"]
    g = ds.groupby(["H", "K", "L"])
    
    result = g[["w", "wI"]].sum()
    result["I"] = result["wI"] / result["w"]
    result["SIGI"] = np.sqrt(1 / result["w"]).astype("Stddev")
    result["N"] = g.size()
    result.merged = True

    # Reshape anomalous data and use to compute IMEAN / SIGIMEAN
    result = result.unstack_anomalous()
    result.loc[:, ["N(+)", "N(-)"]] = result[["N(+)", "N(-)"]].fillna(0).astype("I")
    result["IMEAN"] = result[["wI(+)", "wI(-)"]].sum(axis=1) / result[["w(+)", "w(-)"]].sum(axis=1).astype("Intensity")
    result["SIGIMEAN"] =  np.sqrt(1 / (result[["w(+)", "w(-)"]].sum(axis=1))).astype("Stddev")
    
    # Adjust SIGIMEAN for centric reflections due to duplicated values in
    # Friedel columns
    centrics = result.label_centrics()["CENTRIC"]
    result.loc[centrics, "SIGIMEAN"] *= np.sqrt(2)

    result = result[["IMEAN", "SIGIMEAN", "I(+)", "SIGI(+)", "I(-)", "SIGI(-)", "N(+)", "N(-)"]]

    if sort:
        return result.sort_index()
    
    return result

