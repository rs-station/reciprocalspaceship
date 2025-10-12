import mmap
import re
from typing import Union

import gemmi
import numpy as np

from reciprocalspaceship import DataSet, concat
from reciprocalspaceship.utils import angle_between, eV2Angstroms

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

# See crystFEL API reference here: https://www.desy.de/~twhite/crystfel/reference/stream_8h.html
_block_markers = {
    "geometry": (r"----- Begin geometry file -----", r"----- End geometry file -----"),
    "chunk": (r"----- Begin chunk -----", r"----- End chunk -----"),
    "cell": (r"----- Begin unit cell -----", r"----- End unit cell -----"),
    "peaks": (r"Peaks from peak search", r"End of peak list"),
    "crystal": (r"--- Begin crystal", r"--- End crystal"),
    "reflections": (r"Reflections measured after indexing", r"End of reflections"),
}


class StreamLoader(object):
    """
    An object that loads stream files into rs.DataSet objects in parallel.
    Attributes
    ----------
    block_regex_bytes : dict
        A dictionary of compiled regular expressions that operate on strings
    block_regex : dict
        A dictionary of compiled regular expressions that operate on byte strings
    """

    peak_list_columns = {
        "H": 0,
        "K": 1,
        "L": 2,
        "I": 3,
        "SigI": 4,
        "peak": 5,
        "background": 6,
        "XDET": 7,
        "YDET": 8,
        "s1x": 9,
        "s1y": 10,
        "s1z": 11,
        "ewald_offset": 12,
        "angular_ewald_offset": 13,
        "ewald_offset_x": 14,
        "ewald_offset_y": 15,
        "ewald_offset_z": 16,
    }

    def __init__(self, filename: str, encoding="utf-8"):
        self.filename = filename
        self.encoding = encoding
        self.block_regex = {}
        self.block_regex_bytes = {}

        # Set up all the regular expressions for finding block boundaries
        for k, (beginning, ending) in _block_markers.items():
            self.block_regex[k + "_begin"] = re.compile(beginning)
            self.block_regex[k + "_end"] = re.compile(ending)
            self.block_regex[k] = re.compile(
                f"(?s){beginning}\n(?P<CRYSTAL_BLOCK>.*?)\n{ending}"
            )

            self.block_regex_bytes[k + "_begin"] = re.compile(
                beginning.encode(self.encoding)
            )
            self.block_regex_bytes[k + "_end"] = re.compile(
                ending.encode(self.encoding)
            )
            self.block_regex_bytes[k] = re.compile(
                f"(?s){beginning}\n(?P<CRYSTAL_BLOCK>.*?)\n{ending}".encode(
                    self.encoding
                )
            )

        self.re_abcstar = re.compile("[abc]star =.+\n")
        self.re_photon_energy = re.compile("photon_energy_eV =.+\n")

        self.re_chunk_metadata = {
            "Image filename": re.compile(r"(?<=Image filename: ).+(?=\n)"),
            "Event": re.compile(r"(?<=Event: ).+(?=\n)"),
            "Image serial number:": re.compile(r"(?<=Image serial number: ).+(?=\n)"),
            "indexed_by": re.compile(r"(?<=indexed_by \= ).+(?=\n)"),
            "photon_energy_eV": re.compile(r"(?<=photon_energy_eV \= ).+(?=\n)"),
            "beam_divergence": re.compile(r"(?<=beam_divergence \= ).+(?=\n)"),
            "beam_bandwidth": re.compile(r"(?<=beam_bandwidth \= ).+(?=\n)"),
        }

        self.re_crystal_metadata = {
            "Cell parameters": re.compile(r"(?<=Cell parameters).+(?=\n)"),
            "astar": re.compile(r"(?<=astar = ).+(?=\n)"),
            "bstar": re.compile(r"(?<=bstar = ).+(?=\n)"),
            "cstar": re.compile(r"(?<=cstar = ).+(?=\n)"),
            "lattice_type": re.compile(r"(?<=lattice_type = ).+(?=\n)"),
            "centering": re.compile(r"(?<=centering = ).+(?=\n)"),
            "unique_axis": re.compile(r"(?<=unique_axis = ).+(?=\n)"),
            "profile_radius": re.compile(r"(?<=profile_radius = ).+(?=\n)"),
            "predict_refine/det_shift": re.compile(
                r"(?<=predict_refine/det_shift ).+(?=\n)"
            ),
            "predict_refine/R": re.compile(r"(?<=predict_refine/R ).+(?=\n)"),
            "diffraction_resolution_limit": re.compile(
                r"(?<=diffraction_resolution_limit = ).+(?=\n)"
            ),
            "num_reflections": re.compile(r"(?<=num_reflections = ).+(?=\n)"),
        }

        # TODO: replace these with the faster, non variabled length equivalents
        self.re_crystal = re.compile(
            r"(?s)--- Begin crystal\n(?P<CRYSTAL_BLOCK>.*?)\n--- End crystal"
        )
        self.re_refls = re.compile(
            r"(?s)Reflections measured after indexing\n(?P<REFL_BLOCK>.*?)\nEnd of reflections"
        )

    def extract_target_unit_cell(self) -> Union[list, None]:
        """
        Search the file header for target unit cell parameters.
        """
        header = self.extract_file_header()
        cell = None
        lattice_type = None

        for line in header.split("\n"):
            if line.startswith("a = "):
                idx = 0
            elif line.startswith("b = "):
                idx = 1
            elif line.startswith("c = "):
                idx = 2
            elif line.startswith("al = "):
                idx = 3
            elif line.startswith("be = "):
                idx = 4
            elif line.startswith("ga = "):
                idx = 5
            else:
                idx = None
            if idx is not None:
                if cell is None:
                    cell = [None] * 6
                value = float(line.split()[2])
                cell[idx] = value
            if line.startswith("lattice_type ="):
                lattice_type = line.split()[-1]

        if lattice_type is not None:
            cell = _cell_constraints[lattice_type](cell)
        return cell

    def calculate_average_unit_cell(self) -> gemmi.UnitCell:
        """
        Compute the average of all cell parameters across the file.
        """
        regex = re.compile(rb"Cell parameters .+\n")
        with open(self.filename, "r") as f:
            memfile = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            lines = regex.findall(memfile)
        if len(lines) == 0:
            raise ValueError(
                f"No unit cell parameters were found in the header of {self.filename}"
            )

        cell = np.loadtxt(lines, usecols=[2, 3, 4, 6, 7, 8], dtype="float32").mean(0)
        cell[:3] *= 10.0

        header = self.extract_file_header()
        lattice_type = None

        for line in header.split("\n"):
            if line.startswith("lattice_type ="):
                lattice_type = line.split()[-1]

        if lattice_type is not None:
            cell = _cell_constraints[lattice_type](cell)
        return cell

    def extract_file_header(self) -> str:
        """
        Extract all the data prior to first chunk and return it as a string.
        """
        with open(self.filename, "r") as f:
            memfile = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            match = self.block_regex_bytes["chunk_begin"].search(memfile)
            header = memfile.read(match.start()).decode()
        return header

    @property
    def available_column_names(self) -> list:
        """Keys which can be passed to parallel_read_crystfel to customize the peak list output"""
        return list(self.peak_list_columns.keys())

    @property
    def available_chunk_metadata_keys(self) -> list:
        """Keys which can be passed to parallel_read_crystfel to customize the chunk level metadata"""
        return list(self.re_chunk_metadata.keys())

    @property
    def available_crystal_metadata_keys(self) -> list:
        """Keys which can be passed to parallel_read_crystfel to customize the crystal level metadata"""
        return list(self.re_crystal_metadata.keys())

    def read_crystfel(
        self,
        wavelength=None,
        chunk_metadata_keys=None,
        crystal_metadata_keys=None,
        peak_list_columns=None,
        num_cpus=-1,
        address="local",
    ) -> list:
        """
        Parse a CrystFEL stream file using multiple processors. Parallelization depends on Joblib (https://joblib.readthedocs.io/en/stable/).
        and will not be installed automatically. Users must manually install it prior to calling this method.

        PARAMETERS
        ----------
        wavelength : float
            Override the wavelength with this value. Wavelength is used to compute Ewald offsets.
        chunk_metadata_keys : list
            A list of metadata_keys which will be returned in the resulting dictionaries under the 'chunk_metadata' entry.
            A list of possible keys is stored as stream_loader.available_chunk_metadata_keys
        crytal_metadata_keys : list
            A list of metadata_keys which will be returned in the resulting dictionaries under the 'crystal_metadata' entry.
            A list of possible keys is stored as stream_loader.available_crystal_metadata_keys
        peak_list_columns : list
            A list of columns to include in the peak list numpy arrays.
            A list of possible column names is stored as stream_loader.available_column_names.
        num_cpus : int (optional)
            The number of cpus to use. By default, use all the available cores.
            For more info see the n_jobs parameter [here](https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html)

        RETURNS
        -------
        chunks : list
            A list of dictionaries containing the per-chunk data. The 'peak_lists' item contains a
            numpy array with shape n x 14 with the following information.
                h, k, l, I, SIGI, peak, background, fs/px, ss/px, s1x, s1y, s1z,
                ewald_offset, angular_ewald_offset
        """
        if peak_list_columns is not None:
            peak_list_columns = [self.peak_list_columns[s] for s in peak_list_columns]

        with open(self.filename, "r") as f:
            memfile = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            beginnings_and_ends = zip(
                self.block_regex_bytes["chunk_begin"].finditer(memfile),
                self.block_regex_bytes["chunk_end"].finditer(memfile),
            )
            from joblib import Parallel, delayed

            def parse_chunk(loader: StreamLoader, *args):
                return loader._parse_chunk(*args)

            results = Parallel(num_cpus)(
                delayed(parse_chunk)(
                    self,
                    begin.start(),
                    end.end(),
                    wavelength,
                    chunk_metadata_keys,
                    crystal_metadata_keys,
                    peak_list_columns,
                )
                for begin, end in beginnings_and_ends
            )
        return results

    def _extract_chunk_metadata(self, chunk_text, metadata_keys=None):
        if metadata_keys is None:
            return None
        result = {}
        for k in metadata_keys:
            re = self.re_chunk_metadata[k]
            for v in re.findall(chunk_text):
                result[k] = v
        return result

    def _extract_crystal_metadata(self, xtal_text, metadata_keys=None):
        if metadata_keys is None:
            return None
        result = {}
        for k in metadata_keys:
            re = self.re_crystal_metadata[k]
            for v in re.findall(xtal_text):
                result[k] = v
        return result

    def _parse_chunk(
        self,
        start,
        end,
        wavelength,
        chunk_metadata_keys,
        crystal_metadata_keys,
        peak_list_columns,
    ):
        with open(self.filename, "r") as f:
            f.seek(start)
            data = f.read(end - start)

            if wavelength is None:
                ev_match = self.re_photon_energy.search(data)
                ev_line = data[ev_match.start() : ev_match.end()]
                photon_energy = np.float32(ev_line.split()[2])
                wavelength = eV2Angstroms(photon_energy)
                lambda_inv = np.reciprocal(wavelength)
            else:
                lambda_inv = np.reciprocal(wavelength)

            peak_lists = []
            a_matrices = []
            chunk_metadata = None
            crystal_metadata = []
            header = None
            for xmatch in self.re_crystal.finditer(data):
                xdata = data[xmatch.start() : xmatch.end()]
                if header is None:
                    header = data[: xmatch.start()]

                # crystal_metadata.append(self._extract_crystal_metadata(xdata))
                A = (
                    np.loadtxt(
                        self.re_abcstar.findall(xdata),
                        usecols=[2, 3, 4],
                        dtype="float32",
                    ).T
                    / 10.0
                )
                a_matrices.append(A)

                for pmatch in self.re_refls.finditer(xdata):
                    pdata = xdata[pmatch.start() : pmatch.end()]
                    crystal_metadata.append(
                        self._extract_crystal_metadata(xdata, crystal_metadata_keys)
                    )
                    peak_array = np.loadtxt(
                        pdata.split("\n")[2:-1],
                        usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8),
                        dtype="float32",
                    )
                    s0 = np.array([0, 0, lambda_inv], dtype="float32").T
                    q = (A @ peak_array[:, :3].T).T
                    s1 = q + s0

                    # This is way faster than np.linalg.norm for small dimensions
                    x, y, z = s1.T
                    s1_norm = np.sqrt(x * x + y * y + z * z)
                    ewald_offset = s1_norm - lambda_inv

                    # project calculated s1 onto the ewald sphere
                    s1_obs = lambda_inv * s1 / s1_norm[:, None]

                    # Compute the angular ewald offset
                    q_obs = s1_obs - s0
                    qangle = np.sign(ewald_offset) * angle_between(q, q_obs)

                    peak_array = np.concatenate(
                        (
                            peak_array,
                            s1,
                            ewald_offset[:, None],
                            qangle[:, None],
                            s1_obs - s1,  # Ewald offset vector
                        ),
                        axis=-1,
                    )
                    if peak_list_columns is not None:
                        peak_array = peak_array[:, peak_list_columns]
                    peak_lists.append(peak_array)

        if header is None:
            header = data
        chunk_metadata = self._extract_chunk_metadata(header, chunk_metadata_keys)

        result = {
            "wavelength": wavelength,
            "A_matrices": a_matrices,
            "peak_lists": peak_lists,
        }
        if chunk_metadata_keys is not None:
            result[chunk_metadata_keys] = chunk_metadata
        if crystal_metadata_keys is not None:
            result[crystal_metadata_keys] = crystal_metadata
        return result


def read_crystfel(
    streamfile: str,
    spacegroup=None,
    encoding="utf-8",
    columns=None,
    num_cpus=-1,
    address="local",
) -> DataSet:
    """
    Initialize attributes and populate the DataSet object with data from a CrystFEL stream with indexed reflections.
    This is the output format used by CrystFEL software when processing still diffraction data.

    This method is parallelized across CPUs speed up parsing. Parallelization depends on the Joblib Library.
    and will not be installed automatically. Users must manually install it prior to calling this method.

    Parameters
    ----------
    streamfile : str
        name of a .stream file
    spacegroup : gemmi.SpaceGroup or int or string (optional)
        optionally set the spacegroup of the returned DataSet.
    encoding : str
        The type of byte-encoding (optional, 'utf-8').
    columns : list (optional)
        Optionally specify the columns of the output by a list of strings.
        The default list is: [ "H", "K", "L", "I", "SigI", "BATCH", "s1x", "s1y", "s1z", "ewald_offset", "angular_ewald_offset", "XDET", "YDET" ]
        See `rs.io.crystfel.StreamLoader().available_column_names` for a list of available
        column names and *Notes* for a description of the returned columns
    num_cpus : int (optional)
        By default, the model will use all available cores. For very large cpu counts, this may consume
        too much memory. Decreasing num_cpus may help.

    Returns
    --------
    rs.DataSet

    Notes
    -----
    The following columns are included in the returned DataSet object:

        - H, K, L: Miller indices of each reflection
        - I, SigI: Intensity and associated uncertainty
        - BATCH: Image number
        - s1x, s1y, s1z: scattered beam wavevector which points from the sample to the bragg peak
        - ewald_offset: the distance in cartesian space (1/angstroms) between the observed reflection and the ewald sphere
        - angular_ewald_offset: the distance in polar coordinates (degrees) between the observed reflection and the ewald sphere
        - XDET, YDET: Internal detector panel coordinates
    """
    if not streamfile.endswith(".stream"):
        raise ValueError("Stream file should end with .stream")

    # read data from stream file
    if columns is None:
        columns = [
            "H",
            "K",
            "L",
            "I",
            "SigI",
            "BATCH",
            "s1x",
            "s1y",
            "s1z",
            "ewald_offset",
            "angular_ewald_offset",
            "XDET",
            "YDET",
        ]
    peak_list_columns = [
        i for i in columns if i != "BATCH"
    ]  # BATCH is computed afterward

    mtz_dtypes = {
        "H": "H",
        "K": "H",
        "L": "H",
        "I": "J",
        "SigI": "Q",
        "BATCH": "B",
    }
    for k in columns:
        mtz_dtypes[k] = mtz_dtypes.get(k, "R")

    loader = StreamLoader(streamfile, encoding=encoding)
    cell = loader.extract_target_unit_cell()

    batch = 0
    ds = []

    for chunk in loader.read_crystfel(
        peak_list_columns=peak_list_columns,
        num_cpus=num_cpus,
        address=address,
    ):
        for peak_list in chunk["peak_lists"]:
            _ds = DataSet(
                peak_list,
                columns=peak_list_columns,
                cell=cell,
                spacegroup=spacegroup,
                merged=False,
            )
            _ds["BATCH"] = batch
            ds.append(_ds)
            batch += 1

    ds = concat(ds, axis=0, check_isomorphous=False, copy=False, ignore_index=True)

    mtz_dtypes = {
        "H": "H",
        "K": "H",
        "L": "H",
        "I": "J",
        "SigI": "Q",
        "BATCH": "B",
    }
    for k in ds:
        mtz_dtypes[k] = mtz_dtypes.get(k, "R")

    ds = ds.astype(mtz_dtypes, copy=False)
    ds.set_index(["H", "K", "L"], inplace=True)

    return ds
