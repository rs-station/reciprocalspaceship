import numpy as np

import reciprocalspaceship as rs
from reciprocalspaceship.decorators import inplace


@inplace
def compute_intensity_from_structurefactor(
    ds,
    F_key,
    SigF_key,
    output_columns=None,
    inplace=False,
):
    """
    Compute intensities (I) and uncertainty estimates (SigI) from structure
    factor amplitudes (F) and their uncertainties (SigF) using error propagation

    Intensity computed as I = SigF*SigF + F*F. Intensity error estimate
    approximated as SigI = abs(2*F*SigF)

    Parameters
    ----------
    ds : DataSet
        Input DataSet containing columns with F_key and SigF_key labels
    F_key : str
        Column label for structure factor amplitudes
    SigF_key : str
        Column label for structure factor error estimates
    output_columns : list or tuple of column names
        Column labels to be added to ds for calculated I and
        SigI, respectively. output_columns must have len=2.
    inplace : bool
        Whether to modify the DataSet in place or create a copy

    Returns
    -------
    DataSet
        DataSet with 2 additional columns corresponding to calculated
        intensities and intensity error estimates

    """

    if output_columns is None:
        output_columns = ["I_calc", "SigI_calc"]
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

    # Confirm F_key and SigF_key are of the correct dtype
    if not isinstance(ds.dtypes[F_key], rs.StructureFactorAmplitudeDtype):
        raise ValueError(
            f"Column {F_key} is not of type rs.StructureFactorAmplitudeDtype."
            f"Try again and provide structure factor amplitudes for F_key"
        )
    if not isinstance(ds.dtypes[SigF_key], rs.StandardDeviationDtype):
        raise ValueError(
            f"Column {SigF_key} is not of type rs.StandardDeviationDtype."
            f"Try again and provide standard deviations for SigF_key"
        )

    # Initialize outputs
    ds[I_key] = (ds[F_key] * ds[F_key] + ds[SigF_key] * ds[SigF_key]).astype(
        "Intensity"
    )
    ds[SigI_key] = (2 * ds[F_key] * ds[SigF_key]).abs().astype("Stddev")

    return ds
