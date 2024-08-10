from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import glob
import os
from copy import deepcopy
import numpy as np

from scitbx.matrix import sqr
from cctbx import miller, crystal
from iotbx import mtz
from dxtbx.model import ExperimentList
from dials.array_family import flex


import reciprocalspaceship as rs


def refls_to_q(refls, detector, beam):
    """
    :param refls: reflection table
    :param detector: detector
    :param beam: beam
    """

    orig_vecs = {}
    fs_vecs = {}
    ss_vecs = {}
    u_pids = set(refls['panel'])
    for pid in u_pids:
        orig_vecs[pid] = np.array(detector[pid].get_origin())
        fs_vecs[pid] = np.array(detector[pid].get_fast_axis())
        ss_vecs[pid] = np.array(detector[pid].get_slow_axis())

    s1_vecs = []
    q_vecs = []
    panels = refls["panel"]
    n_refls = len(refls)
    for i_r in range(n_refls):
        r = refls[i_r]
        pid = r['panel']
        i_fs, i_ss, _ = r['xyzobs.px.value']
        panel = detector[pid]
        orig = orig_vecs[pid] #panel.get_origin()
        fs = fs_vecs[pid] #panel.get_fast_axis()
        ss = ss_vecs[pid] #panel.get_slow_axis()

        fs_pixsize, ss_pixsize = panel.get_pixel_size()
        s1 = orig + i_fs*fs*fs_pixsize + i_ss*ss*ss_pixsize  # scattering vector
        s1 = s1 / np.linalg.norm(s1) / beam.get_wavelength()
        s1_vecs.append(s1)
        q_vecs.append(s1-beam.get_s0())

    return np.vstack(q_vecs)


def refls_to_hkl(refls, detector, beam, crystal):
    """
    convert pixel panel reflections to miller index data
    :param refls:  reflecton table for a panel or a tuple of (x,y)
    :param detector:  dxtbx detector model
    :param beam:  dxtbx beam model
    :param crystal: dxtbx crystal model
    """
    if 'rlp' not in list(refls.keys()):
        q_vecs = refls_to_q(refls, detector, beam)
    else:
        q_vecs = np.vstack([refls[i_r]['rlp'] for i_r in range(len(refls))])
    Ai = sqr(crystal.get_A()).inverse()
    Ai = Ai.as_numpy_array()
    HKL = np.dot( Ai, q_vecs.T)
    HKLi = np.ceil(HKL-0.5)
    return np.vstack(HKL).T, np.vstack(HKLi).T


def get_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("dirname", type=str, help="diffBragg.stills_process output folder with integrated.refls")
    parser.add_argument("mtz", type=str, help="output mtz name")
    parser.add_argument("--max", default=2500, type=int, help="max number of refls in a shot (default 2500)")
    parser.add_argument("--ucell", default=None, nargs=6, type=float, help="unit cell params (default will be average experiment crystal)")
    parser.add_argument("--symbol",type=str, default=None)
    parser.add_argument("--batchFromFilename", action="store_true", help="get the batch ID from the filenames assuming it ends in the standard e.g. #####.cbf or #####.mccd ")
    return parser


class ReflData:
    """
    organizes refl attributes
    TODO: __add__ method for MPI reduce
    """

    def __init__(self):
        self.sx , self.sy , self.sz = [],[],[]
        self.x , self.y = [],[]
        self.I , self.sigI = [],[]
        self.h , self.k , self.l = [],[],[]
        self.dh , self.dk , self.dl = [],[],[]
        self.batch , self.dpsi = [],[]
        self.data_attrs = 'sx', 'sy', 'sz', 'x', 'y', 'I', 'sigI', 'h', 'k', 'l', 'dh', 'dk', 'dl', 'batch', 'dpsi'

    def extend(self, other):
        if isinstance(other, ReflData):
            for name in self.data_attrs:
                attr = getattr(self, name)
                attr += getattr(other, name)


def get_paths_inds(fnames, rank=0, size=1):
    """
    :param fnames: integrated reflection table files
    :param rank: process id
    :param size: process pool size
    :return:
    """

    paths_inds = []
    for i,f in enumerate(fnames):
        if i % size != rank:
            continue
        
        expt_f = f.replace(".refl", ".expt")
        assert os.path.exists(expt_f)
        El = ExperimentList.from_file(expt_f, False)
        for E in El:
            # assert paths() and inds() both have length=1
            path = E.imageset.paths()[0]
            ind = E.imageset.indices()[0]
            paths_inds.append( (path, ind))
        if rank==0:
            print("Done finding paths/inds in %s." % f)

    return paths_inds


def get_refl_data(fnames, batch_map, batch_from_filename=False, rank=0, size=1):
    """
    :param fnames: integrated reflection table files
    :param batch_map: images are defined by path (filename) and image indices
        this maps keys are (filename, idx) -> batchNum where batchNum
    :param batch_from_filename: if True, ignore the batch map and retrieve the batch number
        from the standard rotation scan filenames something_[run]_#####.cbf
    :param rank: process id
    :param size: process pool size
    :return:
    """

    reda = ReflData()

    for i_f,f in enumerate(fnames):
        if i_f % size != rank:
            continue

        expt_f = f.replace(".refl", ".expt")
        if rank==0:
            print(f"loading {expt_f} ({i_f+1}/{len(fnames)})...")
        El = ExperimentList.from_file(expt_f, False)

        Rall = flex.reflection_table.from_file(f)
        for i_expt, E in enumerate(El):
            R = Rall.select(Rall['id']==i_expt)
            iset = E.imageset
            path = iset.paths()[0]
            ind = iset.indices()[0]
            batch_num = batch_map[(path,ind)]
            if batch_from_filename:
                try:
                    batch_num = int(iset.paths()[0].split("_")[-1].split(".")[0])
                except Exception as err:
                    print("batchFromFilename failed with", str(err))
                    pass

            H, Hi = refls_to_hkl(R, E.detector, E.beam, E.crystal)

            reda.batch += [batch_num]* len(R)

            hf, kf, lf = H.T
            _h,_k,_l = Hi.T
            reda.dh += list(_h-hf)
            reda.dk += list(_k-kf)
            reda.dl += list(_l-lf)
            reda.dpsi += list(R['delpsical.rad'])
            reda.h += list(_h)
            reda.k += list(_k)
            reda.l += list(_l)

            reda.I += list(R["intensity.sum.value"].as_numpy_array())
            reda.sigI += list(np.sqrt(R["intensity.sum.variance"]))

            _x, _y, _ = R['xyzobs.px.value'].parts()
            reda.x += list(_x)
            reda.y += list(_y)
            
            _sx, _sy, _sz = R['s1'].parts()
            reda.sx += list(_sx)
            reda.sy += list(_sy)
            reda.sz += list(_sz)

    return reda


def reda_to_rs(reda, symbol, ucell, mtz=None):
    """
    :param reda: ReflData instance
    :param symbol: space group symbol e.g. P6
    :param ucell: unit cell tuple (ang,ang,ang,deg,deg,deg)
    :param mtz: mtz filename
    :return: rs dataframe
    """
    symm = crystal.symmetry(ucell, symbol)
    sg_num = symm.space_group().info().type().number()
    ds = rs.DataSet({"H":reda.h, "K":reda.k, "L":reda.l,
        "dH":reda.dh, "dK":reda.dk, "dL":reda.dl,
        "BATCH": reda.batch, "DPSI": reda.dpsi,
        "I": reda.I, "SigI": reda.sigI, "X":reda.x, "Y":reda.y},
        cell=ucell, spacegroup=sg_num)
    ds["SX"] = reda.sx
    ds["SY"] = reda.sy
    ds["SZ"] = reda.sz
    ds["PARTIAL"] = True
    if mtz is not None:
        ds.infer_mtz_dtypes().set_index(["H","K","L"], drop=True).write_mtz(mtz)
        print("Wrote %s." % mtz)
    return ds

