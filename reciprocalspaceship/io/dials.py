import logging

import msgpack
import numpy as np

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
    return vals.T


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
    ds = ds.infer_mtz_dtypes().set_index(["H", "K", "L"], drop=True)
    return ds


@cellify
@spacegroupify
def _get_refl_data(fnames, unitcell, spacegroup, rank=0, size=1):
    """

    Parameters
    ----------
    fnames: integrated refl fioles
    unitcell: unit cell tuple (6 params Ang,Ang,Ang,deg,deg,deg)
    spacegroup: space group name e.g. P4
    rank: process Id [0-N) where N is num proc
    size: total number of proc (N)

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
        _, _, R = msgpack.load(open(f, "rb"), strict_map_key=False)
        refl_data = R["data"]
        expt_id_map = R["identifiers"]
        h, k, l = get_msgpack_data(refl_data, "miller_index")
        (I,) = get_msgpack_data(refl_data, "intensity.sum.value")
        (sigI,) = get_msgpack_data(refl_data, "intensity.sum.variance")
        x, y, _ = get_msgpack_data(refl_data, "xyzcal.px")
        sx, sy, sz = get_msgpack_data(refl_data, "s1")
        (dpsi,) = get_msgpack_data(refl_data, "delpsical.rad")
        (local_id,) = get_msgpack_data(refl_data, "id")
        global_id = np.array([expt_id_map[li] for li in local_id])
        ds = rs.DataSet(
            {
                "H": h,
                "K": k,
                "L": l,
                "BATCH": global_id,
                "DPSI": dpsi,
                "I": I,
                "SigI": sigI,
                "X": x,
                "Y": y,
            },
            cell=unitcell,
            spacegroup=spacegroup,
        )
        ds["SX"] = sx
        ds["SY"] = sy
        ds["SZ"] = sz
        ds["PARTIAL"] = True
        all_ds.append(ds)
    if all_ds:
        all_ds = rs.concat(all_ds)
    else:
        all_ds = None
    return all_ds


def _read_dials_stills_skip_ray(*args, **kwargs):
    """run read_dials_stills without trying to import ray"""
    return _concat(_get_refl_data(*args, **kwargs))


@cellify
@spacegroupify
def read_dials_stills_ray(fnames, unitcell, spacegroup, numjobs=10):
    """

    Parameters
    ----------
    fnames: integration files
    unitcell: unit cell tuple (6 params Ang,Ang,Ang,deg,deg,deg)
    spacegroup: space group name e.g. P4
    numjobs: number of jobs

    Returns
    -------
    RS dataset (pandas Dataframe)
    """
    if not check_for_ray():
        refl_data = _get_refl_data(fnames, unitcell, spacegroup)
    else:
        import ray

        # importing resets log level
        set_ray_loglevel(LOGGER.level)
        ray.init(num_cpus=numjobs, log_to_driver=LOGGER.level == logging.DEBUG)

        # get the refl data
        get_refl_data = ray.remote(_get_refl_data)
        refl_data = ray.get(
            [
                get_refl_data.remote(fnames, unitcell, spacegroup, rank, numjobs)
                for rank in range(numjobs)
            ]
        )

    ds = _concat(refl_data)
    return ds
