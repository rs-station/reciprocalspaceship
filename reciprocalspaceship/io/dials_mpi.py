from mpi4py import MPI

COMM = MPI.COMM_WORLD
from reciprocalspaceship.io import dials


def read_dials_stills_mpi(dirnames, ucell, symbol):
    """

    Parameters
    ----------
    dirnames: folders containing stills process results (integrated.refl)
    ucell: unit cell tuple (6 params Ang,Ang,Ang,deg,deg,deg)
    symbol: space group name e.g. P4

    Returns
    -------
    RS dataset (pandas Dataframe) if MPI rank==0 else None
    """
    fnames = None
    if COMM.rank == 0:
        fnames = dials.get_fnames(dirnames)
    fnames = COMM.bcast(fnames)

    refl_data = dials._get_refl_data(fnames, ucell, symbol, COMM.rank, COMM.size)
    refl_data = COMM.gather(refl_data)
    ds = None
    if COMM.rank == 0:
        ds = dials._concat(refl_data)

    return ds
