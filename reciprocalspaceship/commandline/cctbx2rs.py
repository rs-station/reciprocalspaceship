#!/usr/bin/env python
import glob
import ray


import reciprocalspaceship as rs
from reciprocalspaceship.utils import cctbx_to_rs as to_rs


def main():
    parser = to_rs.get_parser()
    parser.add_argument("--nj", default=10, type=int, help="number of workers!")
    args = parser.parse_args()
    assert args.ucell is not None
    assert args.symbol is not None

    fnames = glob.glob(args.dirname+"/*integrated.refl")
    print("Found %d files" % len(fnames))

    ray.init(num_cpus=args.nj)

    # get the refl data
    get_refl_data = ray.remote(to_rs.get_refl_data)
    refl_data = ray.get( [get_refl_data.remote(fnames, args.ucell, args.symbol, rank, args.nj) \
        for rank in range(args.nj)])
    refl_data = [ds for ds in refl_data if ds is not None]

    print("Combining tables!")
    ds = rs.concat(refl_data)
    expt_ids = set(ds.BATCH)
    print(f"Found {len(ds)} refls from {len(expt_ids)} expts.")
    print("Mapping batch column.")
    expt_id_map = {name:i for i,name in enumerate(expt_ids)}
    ds.BATCH = [expt_id_map[eid] for eid in ds.BATCH]
    
    ds.infer_mtz_dtypes().set_index(["H","K","L"], drop=True).write_mtz(args.mtz)
    print("Wrote %s." % args.mtz)
    print("Done!")


if __name__=="__main__":
    main()
