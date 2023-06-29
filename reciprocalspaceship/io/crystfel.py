from typing import Optional, Union

import gemmi
import numpy as np
import pandas as pd

import reciprocalspaceship as rs
from reciprocalspaceship import DataSet
from reciprocalspaceship.decorators import cellify, spacegroupify
from reciprocalspaceship.utils import angle_between

# See Rupp Table 5-2
_cell_constraints = {
    "triclinic": lambda x: x,
    "orthorhombic": lambda x: [x[0], x[1], x[2], 90.0, 90.0, 90.0],
    "monoclinic": lambda x: [x[0], x[1], x[2], 90.0, x[4], 90.0],
    "hexagonal": lambda x: [
        0.5 * (x[0] + x[1]),
        0.5 * (x[0] + x[1]),
        x[2],
        90.0,
        90.0,
        120.0,
    ],
    "rhombohedral": lambda x: [
        0.5 * (x[0] + x[1]),
        0.5 * (x[0] + x[1]),
        x[2],
        90.0,
        90.0,
        120.0,
    ],
    "cubic": lambda x: [
        np.mean(x[:3]),
        np.mean(x[:3]),
        np.mean(x[:3]),
        90.0,
        90.0,
        90.0,
    ],
    "tetragonal": lambda x: [
        0.5 * (x[0] + x[1]),
        0.5 * (x[0] + x[1]),
        x[2],
        90.0,
        90.0,
        90.0,
    ],
}


class StreamLoader:
    @cellify
    @spacegroupify
    def __init__(
        self,
        stream_file: str,
        cell: Optional[gemmi.UnitCell] = None,
        spacegroup: Optional[gemmi.SpaceGroup] = None,
        wavelength: Optional[float] = None,
    ):
        """
        This class can be used to convert CrystFEL `.stream` files into `rs.DataSet` objects.

        Parameters
        ----------
        stream_file : str
            The path of the `.stream` file. Must end in `.stream`.
        cell : gemmi.UnitCell (optional)
            The cell to assign to the DataSet. If None is supplied, it will be determined from the file.
        spacegroup : gemmi.SpaceGroup (optional)
            The spacegroup to assign to the DataSet.
        wavelength : float (optional)
            The wavelength to use for geometry calculations. If None is supplied, it will be determined from the file.
        """
        super().__init__()
        self.cell = cell
        if self.cell is None:
            self.cell = self.get_target_cell(stream_file)

        self.spacegroup = spacegroup
        self.wavelength = wavelength
        if self.wavelength is None:
            self.wavelength = self.get_wavelength(stream_file)
        self.inverse_wavelength = 1.0 / self.wavelength
        if not stream_file.endswith(".stream"):
            raise ValueError("Stream file should end with .stream")
        self.stream_file = stream_file

    @staticmethod
    def to_crystals(stream_file: str) -> iter:
        """Convert the stream file into an iterator of crystal blocks."""
        return StreamLoader.to_blocks(stream_file, "crystal")

    @staticmethod
    def get_lattice_type(stream_file: str) -> str:
        """Extract the crystal system string from a stream file."""
        lines = next(StreamLoader.to_blocks(stream_file, "cell"))
        for line in lines:
            if line.startswith("lattice_type ="):
                lattice_type = line.split()[2]
                return lattice_type
        raise ValueError("No lattice_type entry!")

    @staticmethod
    def get_target_cell(stream_file: str) -> gemmi.UnitCell:
        """Extract the target unit cell constants from a stream file."""
        lines = next(StreamLoader.to_blocks(stream_file, "cell"))
        cell = [0.0] * 6
        for line in lines:
            if line.startswith("a ="):
                cell[0] = float(line.split()[2])
            elif line.startswith("b ="):
                cell[1] = float(line.split()[2])
            elif line.startswith("c ="):
                cell[2] = float(line.split()[2])
            elif line.startswith("al ="):
                cell[3] = float(line.split()[2])
            elif line.startswith("be ="):
                cell[4] = float(line.split()[2])
            elif line.startswith("ga ="):
                cell[5] = float(line.split()[2])

        return gemmi.UnitCell(*cell)

    @staticmethod
    def get_wavelength(stream_file: str) -> float:
        """Extract the wavelength from a stream file."""
        geo = next(StreamLoader.to_blocks(stream_file, "geometry"))
        for line in geo:
            if line.startswith("photon_energy"):
                eV = float(line.split()[2])
                lam = rs.utils.ev2angstroms(eV)
                return lam

    @staticmethod
    def online_mean_variance(
        iterator,
    ) -> (Union[float, np.ndarray], Union[float, np.ndarray]):
        """Compute the mean and variance of an iterator of floats or arrays online."""

        def update(count, mean, m2, value):
            count = count + 1
            delta = value - mean
            mean += delta / count
            delta2 = value - mean
            m2 += delta * delta2
            return count, mean, m2

        count, mean, m2 = 0, 0, 0
        for value in iterator:
            count, mean, m2 = update(count, mean, m2, value)

        variance = m2 / count
        return mean, variance

    @staticmethod
    def get_average_cell(stream_file, constrain_by_crystal_system=True):
        """
        Gets the average cell across all the indexed
        crystals in a stream file.

        Parameters
        ----------
        stream_file : str
            The path to a CrystFEL stream file.
        constrain_by_crystal_system : bool (optional)
            Whether to apply the appropriate constraints
            for the crystal system described in the stream.
            The default value is True.

        Returns
        -------
        cell : gemmi.UnitCell
        """

        def cell_iter(stream_file):
            for crystal in StreamLoader.to_crystals(stream_file):
                for line in crystal:
                    if line.startswith("Cell parameters"):
                        cell = line.split()
                        cell = np.array(
                            [
                                cell[2],
                                cell[3],
                                cell[4],
                                cell[6],
                                cell[7],
                                cell[8],
                            ],
                            dtype="float32",
                        )
                        cell[:3] = 10.0 * cell[:3]
                        yield cell
                        break

        mean, variance = StreamLoader.online_mean_variance(cell_iter(stream_file))
        cell = gemmi.UnitCell(*mean)
        if constrain:
            lattice_type = StreamLoader.get_lattice_type(stream_file)
            cell = _cell_constraints[lattice_type](cell)
        return cell

    @staticmethod
    def to_blocks(stream_file: str, block_name: str) -> iter:
        """
        Parameters
        ----------
        stream_file : str
            The path to a CrystFEL stream file.
        block_name : str
            One of the following types of blocks
             - 'geometry'
             - 'chunk'
             - 'cell'
             - 'peaks'
             - 'crystal'
             - 'reflections'

        Returns
        -------
        blocks : iter
            An interable containing lists of lines for each block.
        """
        # See crystFEL API reference here: https://www.desy.de/~twhite/crystfel/reference/stream_8h.html
        block_markers = {
            "geometry": (
                "----- Begin geometry file -----",
                "----- End geometry file -----",
            ),
            "chunk": ("----- Begin chunk -----", "----- End chunk -----"),
            "cell": ("----- Begin unit cell -----", "----- End unit cell -----"),
            "peaks": ("Peaks from peak search", "End of peak list"),
            "crystal": ("--- Begin crystal", "--- End crystal"),
            "reflections": (
                "Reflections measured after indexing",
                "End of reflections",
            ),
        }
        block_begin_marker, block_end_marker = block_markers[block_name]

        block = []
        in_block = False
        for line in open(stream_file):
            if line.startswith(block_end_marker):
                in_block = False
                yield block
                block = []
            if in_block:
                block.append(line)
            if line.startswith(block_begin_marker):
                in_block = True

    @staticmethod
    def crystal_to_data(crystal, wavelength, cell=None, dmin=None, batch=None):
        """
        Convert a crystal block (list of strings) to a numpy array.

        Parameters
        ----------
        crystal : list or similar
            A list of strings corresponding to a single crystal
            block in a stream file.
        wavelength : float
            The wavelength of the dataset. Used for calculating ewald
            offsets.
        cell : gemmi.UnitCell (optional)
            The unit cell to use to calculate reflection resolution.
            This is only used for the dmin resolution cutoff.
        dmin : float (optional)
            An optional resolution cutoff. Requires a cell.
        batch : int (optional)
            Optionally supply a batch number which will be appended to
            the output array as another column

        Returns
        -------
        data : array (float32)
            An array of columns which correspond to the following data
             0) H
             1) K
             2) L
             3) I
             4) SigI
             5) s1x
             6) s1y
             7) s1z
             8) ewald_offset
             9) angular_ewald_offset
             10) XDET
             11) YDET
             12) BATCH

        """
        block_name: str
        inverse_wavelength = 1.0 / wavelength
        astar = bstar = cstar = None
        in_refls = False
        crystal_iter = iter(crystal)

        refls = []
        for line in crystal_iter:
            if line.startswith("astar ="):
                astar = (
                    np.array(line.split()[2:5], dtype="float32") / 10.0
                )  # crystfel's notation uses nm-1
            if line.startswith("bstar ="):
                bstar = (
                    np.array(line.split()[2:5], dtype="float32") / 10.0
                )  # crystfel's notation uses nm-1
            if line.startswith("cstar ="):
                cstar = (
                    np.array(line.split()[2:5], dtype="float32") / 10.0
                )  # crystfel's notation uses nm-1
            if line == "End of reflections\n":
                in_refls = False
            if in_refls:
                refls.append(line.split()[:-1])
            if line == "Reflections measured after indexing\n":
                in_refls = True
                crystal_iter = next(crystal_iter)  # skip header

        refls = np.array(refls, dtype="float32")
        hkl = refls[:, :3]

        # Apply dmin
        if dmin is not None:
            if cell is None:
                raise ValueError("dmin supplied without a cell")
            d = cell.calculate_d_array(hkl).astype("float32")
            idx = d >= dmin
            refls = refls[idx]
            d = d[idx]
            hkl = hkl[idx]

        A = np.array([astar, bstar, cstar]).T
        # calculate ewald offset and s1

        s0 = np.array([0, 0, inverse_wavelength]).T
        q = hkl @ A.T  # == (A @ hkl.T).T
        s1 = q + s0
        s1x, s1y, s1z = s1.T
        s1_norm = np.sqrt(s1x * s1x + s1y * s1y + s1z * s1z)

        # project calculated s1 onto the ewald sphere
        s1_obs = inverse_wavelength * s1 / s1_norm[:, None]

        # Compute the ewald offset vector
        eov = s1_obs - s1

        # Compute scalar ewald offset
        eo = s1_norm - inverse_wavelength

        # Compute angular ewald offset
        eo_sign = np.sign(eo)
        q_obs = s1_obs - s0
        ao = eo_sign * rs.utils.angle_between(q, q_obs)

        I = refls[:, 3]
        SigI = refls[:, 4]
        bg = refls[:, 5]

        if batch is not None:
            batch = batch * np.ones_like(I, dtype="int32")
            return np.concatenate(
                (
                    hkl,
                    I[:, None],
                    SigI[:, None],
                    s1,
                    eo[:, None],
                    ao[:, None],
                    refls[:, 7, None],
                    refls[:, 8, None],
                    batch[:, None],
                ),
                axis=1,
            )
        return np.concatenate(
            (
                hkl,
                I[:, None],
                SigI[:, None],
                s1,
                eo[:, None],
                ao[:, None],
                refls[:, 7, None],
                refls[:, 8, None],
            ),
            axis=1,
        )

    def to_dataset(self, spacegroup: Optional[gemmi.SpaceGroup] = None) -> rs.DataSet:
        """Convert self.stream_file to an rs DataSet. Optionally set the spacegroup"""

        def data_gen():
            for i, crystal in enumerate(StreamLoader.to_crystals(self.stream_file)):
                ds = StreamLoader.crystal_to_data(
                    crystal, self.wavelength, self.cell, batch=i + 1
                )
                yield ds

        data = np.concatenate(list(data_gen()), axis=0)

        names = [
            "H",
            "K",
            "L",
            "I",
            "SigI",
            "s1x",
            "s1y",
            "s1z",
            "ewald_offset",
            "angular_ewald_offset",
            "XDET",
            "YDET",
            "BATCH",
        ]
        ds = rs.DataSet(
            data, columns=names, cell=self.cell, spacegroup=spacegroup, merged=False
        ).infer_mtz_dtypes()
        return ds


def read_crystfel(streamfile: str, spacegroup=None) -> DataSet:
    """
    Initialize attributes and populate the DataSet object with data from a CrystFEL stream with indexed reflections.
    This is the output format used by CrystFEL software when processing still diffraction data.

    Parameters
    ----------
    streamfile : str
        name of a .stream file
    spacegroup : gemmi.SpaceGroup or int or string (optional)
        optionally set the spacegroup of the returned DataSet.

    Returns
    --------
    rs.DataSet
    """

    if not streamfile.endswith(".stream"):
        raise ValueError("Stream file should end with .stream")
    loader = StreamLoader(streamfile)
    ds = loader.to_dataset(spacegroup=spacegroup)
    ds.set_index(["H", "K", "L"], inplace=True)
    return ds
