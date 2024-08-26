from reciprocalspaceship.io.ccp4map import write_ccp4_map
from reciprocalspaceship.io.common import check_for_mpi
from reciprocalspaceship.io.crystfel import read_crystfel
from reciprocalspaceship.io.csv import read_csv
from reciprocalspaceship.io.dials import (
    _read_dials_stills_skip_ray,
    read_dials_stills_ray,
)
from reciprocalspaceship.io.mtz import (
    from_gemmi,
    read_cif,
    read_mtz,
    to_gemmi,
    write_mtz,
)
from reciprocalspaceship.io.pickle import read_pickle
from reciprocalspaceship.io.precognition import read_precognition


def read_dials_stills(*args, parallel_backend=None, **kwargs):
    """
    Parameters
    ----------
    fnames: filenames
    unitcell: unit cell tuple, Gemmi unit cell obj
    spacegroup: space group symbol eg P4
    numjobs: (if backend==ray, specify the number of jobs, not needed if backend==mpi)
    parallel_backend: ray, mpi, or None

    Returns
    -------
    rs dataset (pandas Dataframe)
    """
    if parallel_backend not in ["ray", "mpi", None]:
        raise NotImplementedError("parallel_backend should be ray, mpi, or none")

    reader = _read_dials_stills_skip_ray
    if parallel_backend == "ray":
        reader = read_dials_stills_ray
    elif parallel_backend == "mpi":
        if check_for_mpi():
            from reciprocalspaceship.io.dials_mpi import read_dials_stills_mpi as reader
    return reader(*args, **kwargs)
