import reciprocalspaceship as rs
from reciprocalspaceship.utils.asu import in_asu
from reciprocalspaceship.utils.stats import compute_redundancy


def compute_completeness(dataset, bins=10, anomalous=False, dmin=None):
    """
    Compute completeness of DataSet by resolution bin.

    Parameters
    ----------
    dataset : rs.DataSet
        DataSet object to be analyzed
    bins : int
        Number of resolution bins to use
    anomalous : bool
        Whether to compute the anomalous completeness
    dmin : float
        Resolution cutoff to use. If no dmin is supplied, the maximum resolution
        reflection will be used

    Returns
    -------
    rs.DataSet
        DataSet object summarizing the completeness by resolution bin
    """
    # If dataset is merged, all reflections should be in reciprocal ASU
    if dataset.merged:
        if not in_asu(dataset.get_hkls(), dataset.spacegroup).all():
            raise ValueError(
                "Merged DataSet should only contain reflections in the reciprocal ASU"
            )

    if anomalous and dataset.merged:
        H = dataset.stack_anomalous().get_hkls()
    else:
        H = dataset.get_hkls()

    # Compute counts
    h, counts = compute_redundancy(
        H, dataset.cell, dataset.spacegroup, dmin=dmin, anomalous=anomalous
    )
    result = rs.DataSet(
        {"n": counts, "H": h[:, 0], "K": h[:, 1], "L": h[:, 2]},
        spacegroup=dataset.spacegroup,
        cell=dataset.cell,
    )
    result.set_index(["H", "K", "L"], inplace=True)
    result, labels = result.assign_resolution_bins(bins)
    result["observed"] = result["n"] > 0

    # compute overall completeness
    overall = result["observed"].sum() / len(result.index)

    # package results and label with resolution bins
    result = result.groupby("bin")["observed"].agg(["sum", "count"])
    result["completeness"] = result["sum"] / result["count"]
    result = result[["completeness"]]
    result.index = labels
    result.loc["overall", "completeness"] = overall

    return result
