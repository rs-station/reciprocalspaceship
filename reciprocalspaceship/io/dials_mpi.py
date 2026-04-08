from itertools import chain

from reciprocalspaceship.decorators import cellify, spacegroupify
from reciprocalspaceship.io import dials


def mpi_starmap(comm, func, iterable):
    results = []
    for i, item in enumerate(iterable):
        if i % comm.size == comm.rank:
            results.append(func(*item))
    results = comm.gather(results)
    if comm.rank == 0:
        return chain.from_iterable(results)
    return None


@cellify
@spacegroupify
def read_dials_stills_mpi(fnames, unitcell, spacegroup, extra_cols=None, comm=None):
    """

    Parameters
    ----------
    fnames: integrated reflection tables
    unitcell: unit cell tuple (6 params Ang,Ang,Ang,deg,deg,deg)
    spacegroup: space group name e.g. P4
    extra_cols: list of additional column names to read from the refl table
    comm: Optionally override the MPI communicator. The default is MPI.COMM_WORLD with pkl5

    Returns
    -------
    RS dataset (pandas Dataframe) if MPI rank==0 else None
    """
    if comm is None:
        from mpi4py import MPI
        from mpi4py.util import pkl5

        comm = pkl5.Intracomm(MPI.COMM_WORLD)
    ds = mpi_starmap(
        comm,
        dials._get_refl_data,
        ((f, unitcell, spacegroup, extra_cols) for f in fnames),
    )
    return ds


from mpi4py.util import pkl5

files = glob("data/cxidb_81/reflection_data/figure7/*.refl")
comm = pkl5.Intracomm(MPI.COMM_WORLD)  # comm wrapper
ds = rs.io.read_dials_stills(files, numjobs=2, parallel_backend="mpi", comm=comm)
