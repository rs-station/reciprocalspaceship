import glob

import gemmi
import msgpack
import numpy as np
import ray

import reciprocalspaceship as rs

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


def get_fnames(dirnames):
    fnames = []
    for dirname in dirnames:
        fnames += glob.glob(dirname + "/*integrated.refl")
    print("Found %d files" % len(fnames))
    return fnames


def _concat(refl_data):
    refl_data = [ds for ds in refl_data if ds is not None]
    """combine output of _get_refl_data"""
    print("Combining tables!")
    ds = rs.concat(refl_data)
    expt_ids = set(ds.BATCH)
    print(f"Found {len(ds)} refls from {len(expt_ids)} expts.")
    print("Mapping batch column.")
    expt_id_map = {name: i for i, name in enumerate(expt_ids)}
    ds.BATCH = [expt_id_map[eid] for eid in ds.BATCH]
    ds = ds.infer_mtz_dtypes().set_index(["H", "K", "L"], drop=True)
    return ds


def _get_refl_data(fnames, ucell, symbol, rank=0, size=1):
    """

    Parameters
    ----------
    fnames: integrated refl fioles
    ucell: unit cell tuple (6 params Ang,Ang,Ang,deg,deg,deg)
    symbol: space group name e.g. P4
    rank: process Id [0-N) where N is num proc
    size: total number of proc (N)

    Returns
    -------
    RS dataset (pandas Dataframe)

    """

    sg_num = gemmi.find_spacegroup_by_name(symbol).number
    all_ds = []

    for i_f, f in enumerate(fnames):
        if i_f % size != rank:
            continue
        if rank == 0:
            print(f"Loading {i_f+1}/{len(fnames)}")
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
            cell=ucell,
            spacegroup=sg_num,
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


def read_dials_stills(dirnames, ucell, symbol, nj=10):
    """

    Parameters
    ----------
    dirnames: folders containing stills process results (integrated.refl)
    ucell: unit cell tuple (6 params Ang,Ang,Ang,deg,deg,deg)
    symbol: space group name e.g. P4
    nj: number of jobs

    Returns
    -------
    RS dataset (pandas Dataframe)
    """
    fnames = get_fnames(dirnames)
    ray.init(num_cpus=nj)

    # get the refl data
    get_refl_data = ray.remote(_get_refl_data)
    refl_data = ray.get(
        [get_refl_data.remote(fnames, ucell, symbol, rank, nj) for rank in range(nj)]
    )

    ds = _concat(refl_data)
    return ds
