import numpy as np
import gemmi

def write_ccp4_map(realmap, mapfile, cell, spacegroup):
    """
    Write CCP4 map file from NumPy array of real-space density.

    Parameters
    ----------
    realmap : np.ndarray
        3D NumPy array of real-space, voxelized electron density
    mapfile :  str
        Filename to which map will be written
    cell : gemmi.UnitCell
        Unit cell parameters to use in map file
    spacegroup : gemmi.SpaceGroup
        Spacegroup to use in map file
    """
    if not isinstance(realmap, np.ndarray) or not (realmap.ndim == 3):
        raise ValueError("realmap must be a 3-dimension NumPy array")

    # Set up gemmi FloatGrid object with NumPy array
    grid = gemmi.FloatGrid(*realmap.shape)
    grid.set_unit_cell(cell)
    grid.spacegroup = spacegroup
    temp = np.array(grid, copy=False)
    temp[:, :, :] = realmap[:, :, :]

    # Write CCP4 map
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header(2, True)
    ccp4.write_ccp4_map(mapfile)
    
    return
