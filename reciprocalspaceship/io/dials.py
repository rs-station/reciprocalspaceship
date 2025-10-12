import logging
import sys

import msgpack
import numpy as np
import pandas

LOGGER = logging.getLogger("rs.io.dials")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.DEBUG)
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG)
    LOGGER.addHandler(console)

import reciprocalspaceship as rs
from reciprocalspaceship.decorators import cellify, spacegroupify

MSGPACK_DTYPES = {
    "double": np.float64,
    "float": np.float32,
    "int": np.int32,
    "cctbx::miller::index<>": np.int32,
    "vec3<double>": np.float64,
    "std::size_t": np.intp,
}

DEFAULT_COLS = [
    "miller_index",
    "intensity.sum.value",
    "intensity.sum.variance",
    "xyzcal.px",
    "s1",
    "delpsical.rad",
    "id",
]


def _set_logger(verbose):
    level = logging.CRITICAL
    if verbose:
        level = logging.DEBUG

    for log_name in ("rs.io.dials",):
        logger = logging.getLogger(log_name)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


def get_msgpack_data(data, name):
    """

    Parameters
    ----------
    data: msgpack data dict
    name: msgpack data key

    Returns
    -------
    numpy array of values
    """
    dtype, (num, buff) = data[name]
    if dtype in MSGPACK_DTYPES:
        dtype = MSGPACK_DTYPES[dtype]
    else:
        dtype = None  # should we warn here ?
    vals = np.frombuffer(buff, dtype).reshape((num, -1))
    data_dict = {}
    for i, col_data in enumerate(vals.T):
        data_dict[f"{name}.{i}"] = col_data

    # remove the .0 suffix if data is a scalar type
    if len(data_dict) == 1:
        data_dict[name] = data_dict.pop(f"{name}.0")

    return data_dict


def _concat(refl_data):
    """combine output of _get_refl_data"""
    LOGGER.debug("Combining and formatting tables!")
    if isinstance(refl_data, rs.DataSet):
        ds = refl_data
    else:
        refl_data = [ds for ds in refl_data if ds is not None]
        ds = rs.concat(refl_data, check_isomorphous=False)
    expt_ids = set(ds.BATCH)
    LOGGER.debug(f"Found {len(ds)} refls from {len(expt_ids)} expts.")
    LOGGER.debug("Mapping batch column.")
    expt_id_map = {name: i for i, name in enumerate(expt_ids)}
    ds.BATCH = [expt_id_map[eid] for eid in ds.BATCH]
    rename_map = {"miller_index.0": "H", "miller_index.1": "K", "miller_index.2": "L"}
    ds.rename(columns=rename_map, inplace=True)
    LOGGER.debug("Finished combining tables!")
    return ds


def _get_refl_data(fname, unitcell, spacegroup, extra_cols=None):
    """

    Parameters
    ----------
    fname: integrated refl file
    unitcell: gemmi.UnitCell instance
    spacegroup: gemmi.SpaceGroup instance
    extra_cols: list of additional columns to read

    Returns
    -------
    RS dataset (pandas Dataframe)

    """
    LOGGER.debug(f"Loading {fname}")
    pack = _get_refl_pack(fname)
    refl_data = pack["data"]
    expt_id_map = pack["identifiers"]

    if "miller_index" not in refl_data:
        raise IOError("refl table must have a miller_index column")

    ds_data = {}
    col_names = DEFAULT_COLS if extra_cols is None else DEFAULT_COLS + extra_cols
    for col_name in col_names:
        if col_name in refl_data:
            col_data = get_msgpack_data(refl_data, col_name)
            LOGGER.debug(f"... Read in data for {col_name}")
            ds_data = {**col_data, **ds_data}

    if "id" in ds_data:
        ds_data["BATCH"] = np.array([expt_id_map[li] for li in ds_data.pop("id")])
    ds = rs.DataSet(
        ds_data,
        cell=unitcell,
        spacegroup=spacegroup,
    )
    ds["PARTIAL"] = True
    return ds


def _read_dials_stills_serial(fnames, unitcell, spacegroup, extra_cols=None, **kwargs):
    """run read_dials_stills serially"""
    result = [
        _get_refl_data(fname, unitcell, spacegroup, extra_cols) for fname in fnames
    ]
    return result


def _read_dials_stills_joblib(
    fnames, unitcell, spacegroup, num_jobs=10, extra_cols=None
):
    """

    Parameters
    ----------
    fnames: integration files
    unitcell: gemmi.UnitCell instance
    spacegroup: gemmi.SpaceGroup instance
    num_jobs: number of jobs
    extra_cols: list of additional columns to read from refl tables

    Returns
    -------
    RS dataset (pandas Dataframe)
    """
    from joblib import Parallel, delayed

    refl_data = Parallel(num_jobs)(
        delayed(_get_refl_data)(fname, unitcell, spacegroup, extra_cols)
        for fname in fnames
    )
    return refl_data


def dials_to_mtz_dtypes(ds, inplace=True):
    """
    Coerce the dtypes in ds into ones that can be written to an mtz file.
    This will downcast doubles to single precision. If "variance" columns
    are present, they will be converted to "sigma" and assigned
    StandardDeviationDtype.

    Parameters
    ----------
    ds : rs.DataSet
    inplace : bool (optional)
        Convert ds dtypes in place without makeing a copy. Defaults to True.

    Returns
    -------
    ds : rs.DataSet
    """
    rename_map = {}
    for name in ds:
        if "variance" in name:
            new_name = name.replace("variance", "sigma")
            rename_map[name] = new_name
            ds[name] = np.sqrt(ds[name]).astype("Q")
            LOGGER.debug(
                f"Converted column {name} to MTZ-Type Q, took sqrt of the values, and renamed to {new_name}."
            )
    ds.rename(columns=rename_map, inplace=True)
    ds.infer_mtz_dtypes(inplace=True)
    return ds


@cellify
@spacegroupify
def read_dials_stills(
    fnames,
    unitcell=None,
    spacegroup=None,
    numjobs=10,
    parallel_backend=None,
    extra_cols=None,
    verbose=False,
    comm=None,
    mtz_dtypes=False,
):
    """
    Read reflections from still images processed by DIALS from fnames and return
    them as a DataSet. By default, this function will not convert the data from
    dials into an MTZ compatible format.

    Parameters
    ----------
    fnames : list or tuple or string
        A list or tuple of filenames (strings) or a single filename.
    unitcell : gemmi.UnitCell or similar (optional)
        The unit cell assigned to the returned dataset.
    spacegroup : gemmi.SpaceGroup or similar (optional)
        The spacegroup assigned to the returned dataset.
    numjobs : int
        If backend==joblib, specify the number of jobs (ignored if backend==mpi).
    parallel_backend : string (optional)
        "joblib", "mpi", or None for serial.
    extra_cols : list (optional)
        Optional list of additional column names to extract from the refltables. By default, this method will search for
        miller_index, id, s1, xyzcal.px, intensity.sum.value, intensity.sum.variance, delpsical.rad
    verbose : bool
        Whether to print logging info to stdout
    comm : mpi4py.MPI.Comm
        Optionally override the communicator used by backend='mpi'
    mtz_dtypes : bool (optional)
        Optionally convert columns to mtz compatible dtypes. Note this will downcast double precision (64-bit)
        floats to single precision (32-bit).

    Returns
    -------
    ds : rs.DataSet
        The dataset containing reflection info aggregated from fnames. This method will not convert any of the
        columns to native rs MTZ dtypes. DIALS data are natively double precision (64-bit). Converting to MTZ
        will downcast them to 32-bit. Use ds.infer_mtz_dtypes() to convert to native rs dtypes if required.
    """
    _set_logger(verbose)
    if isinstance(fnames, str):
        fnames = [fnames]

    if parallel_backend not in ["joblib", "mpi", None]:
        raise NotImplementedError("parallel_backend should be joblib, mpi, or none")

    kwargs = {
        "fnames": fnames,
        "unitcell": unitcell,
        "spacegroup": spacegroup,
        "extra_cols": extra_cols,
    }
    reader = _read_dials_stills_serial
    if parallel_backend == "joblib":
        kwargs["num_jobs"] = numjobs
        reader = _read_dials_stills_joblib
    elif parallel_backend == "mpi":
        from reciprocalspaceship.io.common import check_for_mpi

        if check_for_mpi():
            from reciprocalspaceship.io.dials_mpi import read_dials_stills_mpi as reader

            kwargs["comm"] = comm
    result = reader(**kwargs)
    if result is not None:
        result = _concat(result)
    if mtz_dtypes:
        dials_to_mtz_dtypes(result, inplace=True)
    return result


def _get_refl_pack(filename):
    pack = msgpack.load(open(filename, "rb"), strict_map_key=False)
    try:
        assert len(pack) == 3
        _, _, pack = pack
    except (TypeError, AssertionError):
        raise IOError("File does not appear to be dials::af::reflection_table")
    return pack


def print_refl_info(reflfile):
    """print contents of `fname`, a reflection table file saved with DIALS"""
    pack = _get_refl_pack(reflfile)
    if "identifiers" in pack:
        idents = pack["identifiers"]
        print(f"\nFound {len(idents)} experiment identifiers in {reflfile}:")
        for i, ident in idents.items():
            print(f"\t{i}: {ident}")
    if "data" in pack:
        data = pack["data"]
        columns = []
        col_space = 0
        for name in data:
            dtype, (_, buff) = data[name]
            columns.append((name, dtype))
            col_space = max(len(dtype), len(name), col_space)
        names, dtypes = zip(*columns)
        df = pandas.DataFrame({"names": names, "dtypes": dtypes})
        print(
            "\nReflection contents:\n"
            + df.to_string(index=False, col_space=col_space + 5, justify="center")
        )

    if "nrows" in pack:
        print(f"\nNumber of reflections: {pack['nrows']} \n")
