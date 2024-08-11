from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import glob
import os
from copy import deepcopy
import numpy as np
import msgpack
import gemmi


import reciprocalspaceship as rs
    
MSGPACK_DTYPES = {
    'double': np.float64,
    'float': np.float32,
    'int': np.int32,
    'cctbx::miller::index<>': np.int32,
    'vec3<double>': np.float64,
    'std::size_t': np.intp}


def get_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("dirname", type=str, help="diffBragg.stills_process output folder with integrated.refls")
    parser.add_argument("mtz", type=str, help="output mtz name")
    parser.add_argument("--ucell", default=None, nargs=6, type=float, help="unit cell params (default will be average experiment crystal)")
    parser.add_argument("--symbol",type=str, default=None)
    return parser


def get_msgpack_data(data, name):
    dtype, (num, buff) = data[name]
    if dtype in MSGPACK_DTYPES:
        dtype = MSGPACK_DTYPES[dtype]
    else:
        dtype=None
    vals = np.frombuffer(buff, dtype).reshape((num, -1))
    return vals.T


def get_refl_data(fnames, ucell, symbol, rank=0, size=1):
    """
    :param fnames: integrated reflection table files
    :ucell: 6 unit cell constants (ang,ang,ang,deg,deg,deg)
    :symbol: space group symbol (e.g. P6)
    :param rank: process id
    :param size: process pool size
    :return:
    """

    sg_num = gemmi.find_spacegroup_by_name(symbol).number
    all_ds = []

    for i_f,f in enumerate(fnames):
        if i_f % size != rank:
            continue
        if rank==0:
            print(f"Loading {i_f+1}/{len(fnames)}")
        _,_,R = msgpack.load(open(f, 'rb'), strict_map_key=False)
        refl_data = R['data']
        expt_id_map = R['identifiers']
        col_names = list(refl_data.keys())
        h,k,l = get_msgpack_data(refl_data, 'miller_index')
        I, = get_msgpack_data(refl_data, 'intensity.sum.value')
        sigI, = get_msgpack_data(refl_data, 'intensity.sum.variance')
        x,y,_ = get_msgpack_data(refl_data, 'xyzcal.px')
        sx,sy,sz = get_msgpack_data(refl_data, 's1')
        dpsi, = get_msgpack_data(refl_data, 'delpsical.rad')
        local_id, = get_msgpack_data(refl_data, 'id')
        global_id = np.array([expt_id_map[li] for li in local_id])
        ds = rs.DataSet({"H":h, "K":k, "L":l,
                "BATCH": global_id, "DPSI": dpsi,
                "I": I, "SigI": sigI, "X":x, "Y":y},
                cell=ucell, spacegroup=sg_num)
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

