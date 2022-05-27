import warnings

import numpy as np

import reciprocalspaceship as rs


def compute_intensity_from_structurefactor(
    ds,
    F_key,
    SigF_key,
    output_columns=None,
    dropna=True,
    inplace=False,
):
    """
    Back-calculate intensities and approximate intensity error estimates from
    structure factor amplitudes and error estimates

    Intensity computed as I = SigF*SigF + F*F. Intensity error estimate
    approximated as SigI = abs(2*F*SigF)

    Parameters
    ----------
    ds : DataSet
        Input DataSet containing columns with F_key and SigF_key labels
    F_key : str
        Column label for structure factor amplitudes to use in back-calculation
    SigF_key : str
        Column label for structure factor error estimates to use in
        back-calculation
    output_columns : list or tuple of column names
        Column labels to be added to ds for recording back-calculated I and
        SigI, respectively. output_columns must have len=2.
    dropna : bool
        Whether to drop reflections with NaNs in F_key or SigF_key
        columns
    inplace : bool
        Whether to modify the DataSet in place or create a copy

    Returns
    -------
    DataSet
        DataSet with 2 additional columns corresponding to back-calculated
        intensities and intensity error estimates

    """

    if not inplace:
        ds = ds.copy()

    # Sanitize input or check for invalid reflections
    if dropna:
        ds.dropna(subset=[F_key, SigF_key], inplace=True)
    else:
        if ds[[F_key, SigF_key]].isna().to_numpy().any():
            raise ValueError(
                f"Input {ds.__class__.__name__} contains NaNs "
                f"in columns '{F_key}' and/or '{SigF_key}'. "
                f"Please fix these input values, or run with dropna=True"
            )

    if output_columns is None:
        output_columns = ["I_back", "SigI_back"]
    (
        I_key,
        SigI_key,
    ) = output_columns

    if I_key in ds.columns:
        raise ValueError(
            f"Input {ds.__class__.__name__} already contains column '{I_key}'."
            f"Try again and use the output_columns argument to pick a new"
            f"output column name."
        )
    if SigI_key in ds.columns:
        raise ValueError(
            f"Input {ds.__class__.__name__} already contains column '{SigI_key}'."
            f"Try again and use the output_columns argument to pick a new"
            f"output column name."
        )

    # Initialize outputs
    ds[I_key] = 0.0
    ds[SigI_key] = 0.0

    F_np = ds[F_key].to_numpy()
    SigF_np = ds[SigF_key].to_numpy()

    I = SigF_np * SigF_np + F_np * F_np
    SigI = np.abs(2 * F_np * SigF_np)

    ds[I_key] = rs.DataSeries(I, index=ds.index, dtype="Intensity")
    ds[SigI_key] = rs.DataSeries(SigI, index=ds.index, dtype="Stddev")

    return ds
