import numpy as np
import pandas as pd

import reciprocalspaceship as rs
from reciprocalspaceship.utils.asu import in_asu
from reciprocalspaceship.utils.binning import assign_with_binedges, bin_by_percentile
from reciprocalspaceship.utils.stats import compute_redundancy


def compute_completeness(
    dataset, bins=10, anomalous="auto", dmin=None, unobserved_value=np.nan
):
    """
    Compute completeness of DataSet by resolution bin.

    This function computes the completeness of a given unmerged or merged
    DataSet. If an unmerged DataSet is provided, the function will compute
    the completeness in the DataSet's spacegroup, regardless of whether the
    reflections are specified in the reciprocal ASU or in P1. If a merged
    DataSet is provided, the reflections must be specified in the reciprocal
    ASU.

    There are several types of completeness that can be computed using this
    function:

    - **completeness (all)**: completeness of data before merging Friedel pairs
      (all reflections in +/- ASU)
    - **completeness (non-anomalous)**: completeness after merging Friedel pairs
      (all reflections in +ASU)
    - **completeness (anomalous)**: completeness of the anomalous data. Only
      accounts for acentric Bijvoet mates measured in both +/- ASU. Centric
      reflections do not factor into this calculation.

    Notes
    -----
    - If the `anomalous` flag is 'auto', it will be auto set to `True` if
      the input DataSet is unmerged or contains columns with Friedel dtypes.
    - If `anomalous=False`, the completeness (non-anomalous) is computed.
    - If `anomalous=True`, all three completeness metrics are
    - `unobserved_value` is only used if `anomalous=True`, and will be used
      for filtering any Friedel observations in 1-col anomalous mode with the
      given value. This is only applied to merged DataSets because unmerged
      data should not use fill values.
    - MTZ files from sources such as phenix.refine may have additional filled
      columns that do not reflect data completeness. Pre-filtering may be
      required in such cases to only include "obs" suffixed columns in order
      to get the desired results.
    - If `anomalous=True` and a merged DataSet is provided, MTZInt data columns
      are removed to avoid potential issues with `unobserved_value` filtering.
      For example, this avoids issues with R-free flags in cases when
      `unobserved_value=0`, which can be the case for aimless output.

    Parameters
    ----------
    dataset : rs.DataSet
        DataSet object to be analyzed
    bins : int
        Number of resolution bins to use
    anomalous : bool or 'auto'
        Whether to compute the anomalous completeness
    dmin : float
        Resolution cutoff to use. If no dmin is supplied, the maximum
        resolution reflection will be used
    unobserved_value : float
        Value of unobserved Friedel mates in `dataset`. Will be used if `anomalous=True`
        for removing unobserved reflections from merged DataSet objects.

    Returns
    -------
    rs.DataSet
        DataSet object summarizing the completeness by resolution bin
    """
    if anomalous == "auto":
        if not isinstance(dataset.merged, bool):
            raise AttributeError(
                f"DataSet.merged should be True or False -- value is: {dataset.merged}"
            )
        # Merged
        elif dataset.merged:
            if any([dataset[c].dtype.is_friedel_dtype() for c in dataset]):
                anomalous = True
            else:
                anomalous = False
        # Unmerged
        else:
            anomalous = True

    # Get bin edges for resolution bins
    dHKL = dataset.compute_dHKL()["dHKL"]
    dmax = dHKL.max()
    if dmin:
        dataset = dataset.loc[dHKL >= dmin]
        dHKL = dHKL[dHKL > dmin]
    assignments, labels, binedges = bin_by_percentile(
        dHKL, bins=bins, ascending=False, return_edges=True
    )

    # Adjust high resolution bin to dmin
    if dmin:
        binedges[-1] = dmin
        fields = labels[-1].split()
        labels[-1] = f"{fields[0]} - {dmin:.2f}"

    # If dataset is merged, all reflections should be in reciprocal ASU
    if dataset.merged:
        if not in_asu(dataset.get_hkls(), dataset.spacegroup).all():
            raise ValueError(
                "Merged DataSet should only contain reflections in the reciprocal ASU"
            )

        if anomalous:
            dataset = dataset[[c for c in dataset if dataset[c].dtype.mtztype != "I"]]
            H = (
                dataset.stack_anomalous()
                .replace(unobserved_value, np.nan)
                .dropna()
                .get_hkls()
            )
        else:
            H = dataset.get_hkls()

    # If unmerged, map to reciprocal ASU
    elif not dataset.merged:
        H = dataset.hkl_to_asu(anomalous=anomalous).get_hkls()

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
    dHKL = result.compute_dHKL()["dHKL"]
    result = result.loc[dHKL < dmax]
    dHKL = dHKL[dHKL < dmax]
    assignments = assign_with_binedges(dHKL, binedges, right_inclusive=False)
    result["bin"] = assignments
    result["observed"] = result["n"] > 0
    asu = result.hkl_to_asu()

    def completeness_by_bin(ds, labels, column="observed"):
        """Compute completeness by bin using given column name"""
        result = ds.groupby("bin")[column].agg(["sum", "count"])
        result["completeness"] = result["sum"] / result["count"]
        result = result[["completeness"]]
        result.index = labels

        overall = ds[column].sum() / len(ds)
        result.loc["overall", "completeness"] = overall
        return result

    # Compute non-anomalous completeness
    comp_nonanom = asu.groupby(["H", "K", "L"])["observed"].any().to_frame()
    comp_nonanom["bin"] = result.loc[comp_nonanom.index, "bin"]

    result = completeness_by_bin(comp_nonanom, labels)
    result.columns = pd.MultiIndex.from_product([result.columns, ["non-anomalous"]])

    if anomalous:

        # Compute completeness (all)
        result2 = completeness_by_bin(asu, labels)
        result2.columns = pd.MultiIndex.from_product([result2.columns, ["all"]])

        # Compute anomalous completeness
        result3 = completeness_by_bin(asu.acentrics, labels)
        result3.columns = pd.MultiIndex.from_product([result3.columns, ["anomalous"]])

        return rs.concat([result2, result, result3], axis=1, check_isomorphous=False)

    return result
