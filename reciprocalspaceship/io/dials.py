import logging

import msgpack
import numpy as np
import pandas

LOGGER = logging.getLogger("rs.io.dials")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    LOGGER.addHandler(console)

import reciprocalspaceship as rs
from reciprocalspaceship.decorators import cellify, spacegroupify
from reciprocalspaceship.io.common import check_for_ray, set_ray_loglevel

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

REQ_COLS = {"miller_index", "id"}


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
        dtype = None
    vals = np.frombuffer(buff, dtype).reshape((num, -1))
    data_dict = {}
    for i, col_data in enumerate(vals.T):
        data_dict[f"{name}.{i}"] = col_data
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
        ds = rs.concat(refl_data)
    expt_ids = set(ds.BATCH)
    LOGGER.debug(f"Found {len(ds)} refls from {len(expt_ids)} expts.")
    LOGGER.debug("Mapping batch column.")
    expt_id_map = {name: i for i, name in enumerate(expt_ids)}
    ds.BATCH = [expt_id_map[eid] for eid in ds.BATCH]
    rename_map = {"miller_index.0": "H", "miller_index.1": "K", "miller_index.2": "L"}
    for name in list(ds):
        if "variance" in name:
            rename_map[name] = name.replace("variance", "sigma")
            ds[name] = np.sqrt(ds[name]).astype("Q")
    ds.rename(columns=rename_map, inplace=True)

    ds = ds.infer_mtz_dtypes().set_index(["H", "K", "L"], drop=True)
    return ds


@cellify
@spacegroupify
def _get_refl_data(fnames, unitcell, spacegroup, rank=0, size=1, extra_cols=None):
    """

    Parameters
    ----------
    fnames: integrated refl fioles
    unitcell: unit cell tuple (6 params Ang,Ang,Ang,deg,deg,deg)
    spacegroup: space group name e.g. P4
    rank: process Id [0-N) where N is num proc
    size: total number of proc (N)
    extra_cols: list of additional columns to read

    Returns
    -------
    RS dataset (pandas Dataframe)

    """

    all_ds = []

    for i_f, f in enumerate(fnames):
        if i_f % size != rank:
            continue

        if rank == 0:
            LOGGER.debug(f"Loading {i_f+1}/{len(fnames)}")
        pack = _get_refl_pack(f)
        refl_data = pack["data"]
        expt_id_map = pack["identifiers"]

        if "miller_index" not in refl_data:
            raise IOError("refl table must have a miller_index column")

        ds_data = {}
        col_names = DEFAULT_COLS if extra_cols is None else DEFAULT_COLS + extra_cols
        for col_name in col_names:
            if col_name in refl_data:
                col_data = get_msgpack_data(refl_data, col_name)
                ds_data = {**col_data, **ds_data}

        if "id" in ds_data:
            ds_data["BATCH"] = np.array([expt_id_map[li] for li in ds_data.pop("id")])
        ds = rs.DataSet(
            ds_data,
            cell=unitcell,
            spacegroup=spacegroup,
        )
        ds["PARTIAL"] = True
        all_ds.append(ds)
    if all_ds:
        all_ds = rs.concat(all_ds)
    else:
        all_ds = None
    return all_ds


def _read_dials_stills_serial(*args, **kwargs):
    """run read_dials_stills without trying to import ray"""
    return _concat(_get_refl_data(*args, **kwargs))


@cellify
@spacegroupify
def read_dials_stills_ray(fnames, unitcell, spacegroup, numjobs=10, extra_cols=None):
    """

    Parameters
    ----------
    fnames: integration files
    unitcell: unit cell tuple (6 params Ang,Ang,Ang,deg,deg,deg)
    spacegroup: space group name e.g. P4
    numjobs: number of jobs
    extra_cols: list of additional columns to read from refl tables

    Returns
    -------
    RS dataset (pandas Dataframe)
    """
    if not check_for_ray():
        refl_data = _get_refl_data(fnames, unitcell, spacegroup, extra_cols=extra_cols)
    else:
        import ray

        # importing resets log level
        set_ray_loglevel(LOGGER.level)
        ray.init(num_cpus=numjobs, log_to_driver=LOGGER.level == logging.DEBUG)

        # get the refl data
        get_refl_data = ray.remote(_get_refl_data)
        refl_data = ray.get(
            [
                get_refl_data.remote(
                    fnames, unitcell, spacegroup, rank, numjobs, extra_cols
                )
                for rank in range(numjobs)
            ]
        )

    ds = _concat(refl_data)
    return ds


@cellify
@spacegroupify
def read_dials_stills(
    fnames, unitcell, spacegroup, numjobs=10, parallel_backend=None, extra_cols=None
):
    """
    Parameters
    ----------
    fnames: filenames
    unitcell: unit cell tuple, Gemmi unit cell obj
    spacegroup: space group symbol eg P4
    numjobs: if backend==ray, specify the number of jobs (ignored if backend==mpi)
    parallel_backend: ray, mpi, or None
    extra_cols: list of additional column names to extract from the refltables. By default, this method will search for
        miller_index, id, s1, xyzcal.px, intensity.sum.value, intensity.sum.variance, delpsical.rad

    Returns
    -------
    rs dataset (pandas Dataframe)
    """
    if parallel_backend not in ["ray", "mpi", None]:
        raise NotImplementedError("parallel_backend should be ray, mpi, or none")

    kwargs = {
        "fnames": fnames,
        "unitcell": unitcell,
        "spacegroup": spacegroup,
        "extra_cols": extra_cols,
    }
    reader = _read_dials_stills_serial
    if parallel_backend == "ray":
        kwargs["numjobs"] = numjobs
        reader = read_dials_stills_ray
    elif parallel_backend == "mpi":
        from reciprocalspaceship.io.common import check_for_mpi

        if check_for_mpi():
            from reciprocalspaceship.io.dials_mpi import read_dials_stills_mpi as reader
    return reader(**kwargs)


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
