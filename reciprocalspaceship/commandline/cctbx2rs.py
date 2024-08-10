#!/usr/bin/env python
import glob
import ray


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
    get_paths_inds = ray.remote(to_rs.get_paths_inds)
    paths_inds = ray.get( [get_paths_inds.remote(fnames, rank, args.nj) for rank in range(args.nj) ])

    # flatten fnames and paths_inds
    paths_inds = [pi for pis in paths_inds for pi in pis]

    print("Found data for %d unique images" % len(paths_inds))

    # create batch mapper:
    batch_map = {path_ind: i for i, path_ind in enumerate(paths_inds)}

    # get the refl data
    get_refl_data = ray.remote(to_rs.get_refl_data)
    refl_data = ray.get( [get_refl_data.remote(fnames, batch_map, args.batchFromFilename, rank, args.nj) \
        for rank in range(args.nj)])

    reda = to_rs.ReflData()
    for other in refl_data:
        reda.extend(other)
        print("Combining refl data from all processes (%d total refls)" % len(reda.h), end="\r", flush=True)
    print("\nDone combining!")
            
    rs = to_rs.reda_to_rs(reda, args.symbol, args.ucell, args.mtz)
    print("Done!")


if __name__=="__main__":
    main()
