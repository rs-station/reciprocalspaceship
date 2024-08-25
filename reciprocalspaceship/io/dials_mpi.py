from mpi4py import MPI

COMM = MPI.COMM_WORLD
from reciprocalspaceship.decorators import cellify, spacegroupify
from reciprocalspaceship.io import dials


@cellify
@spacegroupify
def read_dials_stills_mpi(fnames, unitcell, spacegroup):
    """

    Parameters
    ----------
    fnames: integrated reflection tables
    unitcell: unit cell tuple (6 params Ang,Ang,Ang,deg,deg,deg)
    spacegroup: space group name e.g. P4

    Returns
    -------
    RS dataset (pandas Dataframe) if MPI rank==0 else None
    """

    refl_data = dials._get_refl_data(fnames, unitcell, spacegroup, COMM.rank, COMM.size)
    refl_data = COMM.gather(refl_data)
    ds = None
    if COMM.rank == 0:
        ds = dials._concat(refl_data)

    return ds
