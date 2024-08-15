import gemmi

from reciprocalspaceship import DataSet
from reciprocalspaceship.dtypes.base import MTZDtype
from reciprocalspaceship.utils import in_asu


def from_gemmi(gemmi_mtz):
    """
    Construct DataSet from gemmi.Mtz object

    If the gemmi.Mtz object contains an M/ISYM column and contains duplicated
    Miller indices, an unmerged DataSet will be constructed. The Miller indices
    will be mapped to their observed values, and a partiality flag will be
    extracted and stored as a boolean column with the label, ``PARTIAL``.
    Otherwise, a merged DataSet will be constructed.

    If columns are found with the ``MTZInt`` dtype and are labeled ``PARTIAL``
    or ``CENTRIC``, these will be interpreted as boolean flags used to
    label partial or centric reflections, respectively.

    Parameters
    ----------
    gemmi_mtz : gemmi.Mtz
        gemmi Mtz object

    Returns
    -------
    rs.DataSet
    """
    dataset = DataSet(spacegroup=gemmi_mtz.spacegroup, cell=gemmi_mtz.cell)

    # Build up DataSet
    for c in gemmi_mtz.columns:
        dataset[c.label] = c.array
        # Special case for CENTRIC and PARTIAL flags
        if c.type == "I" and c.label in ["CENTRIC", "PARTIAL"]:
            dataset[c.label] = dataset[c.label].astype(bool)
        else:
            dataset[c.label] = dataset[c.label].astype(c.type)
    dataset.set_index(["H", "K", "L"], inplace=True)

    # Handle unmerged DataSet. Raise ValueError if M/ISYM column is not unique
    m_isym = dataset.get_m_isym_keys()
    if m_isym and dataset.index.duplicated().any():
        if len(m_isym) == 1:
            dataset.merged = False
            dataset.hkl_to_observed(m_isym[0], inplace=True)
        else:
            raise ValueError(
                "Only a single M/ISYM column is supported for unmerged data"
            )
    else:
        dataset.merged = True

    return dataset


def to_gemmi(
    dataset,
    skip_problem_mtztypes,
    project_name,
    crystal_name,
    dataset_name,
):
    """
    Construct gemmi.Mtz object from DataSet

    If ``dataset.merged == False``, the reflections will be mapped to the
    reciprocal space ASU, and a M/ISYM column will be constructed.

    If boolean flags with the label ``PARTIAL`` or ``CENTRIC`` are found
    in the DataSet, these will be cast to the ``MTZInt`` dtype, and included
    in the gemmi.Mtz object.

    Parameters
    ----------
    dataset : rs.DataSet
        DataSet object to convert to gemmi.Mtz
    skip_problem_mtztypes : bool
        Whether to skip columns in DataSet that do not have specified
        mtz datatypes
    project_name : str
        Project name to assign to MTZ file
    crystal_name : str
        Crystal name to assign to MTZ file
    dataset_name : str
        Dataset name to assign to MTZ file

    Returns
    -------
    gemmi.Mtz
    """
    # Check that cell and spacegroup are defined
    if not dataset.cell:
        raise AttributeError(
            f"Instance of type {dataset.__class__.__name__} has no unit cell information"
        )
    if not dataset.spacegroup:
        raise AttributeError(
            f"Instance of type {dataset.__class__.__name__} has no space group information"
        )

    # Check project_name, crystal_name, and dataset_name are str
    if not isinstance(project_name, str):
        raise ValueError(
            f"project_name must be a string. Given type: {type(project_name)}"
        )
    if not isinstance(crystal_name, str):
        raise ValueError(
            f"crystal_name must be a string. Given type: {type(crystal_name)}"
        )
    if not isinstance(dataset_name, str):
        raise ValueError(
            f"dataset_name must be a string. Given type: {type(dataset_name)}"
        )

    # Build up a gemmi.Mtz object
    mtz = gemmi.Mtz()
    mtz.cell = dataset.cell
    mtz.spacegroup = dataset.spacegroup

    # Handle Unmerged data
    if not dataset.merged:
        all_in_asu = in_asu(dataset.get_hkls(), dataset.spacegroup).all()
        if not all_in_asu:
            dataset.hkl_to_asu(inplace=True)

    # Add Dataset with indicated names
    mtz.add_dataset("reciprocalspaceship")
    mtz.datasets[0].project_name = project_name
    mtz.datasets[0].crystal_name = crystal_name
    mtz.datasets[0].dataset_name = dataset_name

    # Construct data for Mtz object
    # GH#255: DataSet is provided using the range_indexed decorator
    columns = []
    for c in dataset.columns:
        cseries = dataset[c]
        if isinstance(cseries.dtype, MTZDtype):
            mtz.add_column(label=c, type=cseries.dtype.mtztype)
            columns.append(c)
        # Special case for CENTRIC and PARTIAL flags
        elif cseries.dtype.name == "bool" and c in ["CENTRIC", "PARTIAL"]:
            mtz.add_column(label=c, type="I")
            columns.append(c)
        elif skip_problem_mtztypes:
            continue
        else:
            raise ValueError(
                f"column {c} of type {cseries.dtype} cannot be written to an MTZ file. "
                f"To skip columns without explicit MTZ dtypes, set skip_problem_mtztypes=True"
            )
    mtz.set_data(dataset[columns].to_numpy(dtype="float32"))

    # Handle Unmerged data
    if not dataset.merged and not all_in_asu:
        dataset.hkl_to_observed(m_isym="M/ISYM", inplace=True)

    return mtz


def read_mtz(mtzfile):
    """
    Populate the dataset object with data from an MTZ reflection file.

    If the gemmi.Mtz object contains an M/ISYM column and contains duplicated
    Miller indices, an unmerged DataSet will be constructed. The Miller indices
    will be mapped to their observed values, and a partiality flag will be
    extracted and stored as a boolean column with the label, ``PARTIAL``.
    Otherwise, a merged DataSet will be constructed.

    If columns are found with the ``MTZInt`` dtype and are labeled ``PARTIAL``
    or ``CENTRIC``, these will be interpreted as boolean flags used to
    label partial or centric reflections, respectively.

    Parameters
    ----------
    mtzfile : str or file
        name of an mtz file or a file object

    Returns
    -------
    DataSet
    """
    gemmi_mtz = gemmi.read_mtz_file(mtzfile)
    return from_gemmi(gemmi_mtz)


def write_mtz(
    dataset,
    mtzfile,
    skip_problem_mtztypes,
    project_name,
    crystal_name,
    dataset_name,
):
    """
    Write an MTZ reflection file from the reflection data in a DataSet.

    If ``dataset.merged == False``, the reflections will be mapped to the
    reciprocal space ASU, and a M/ISYM column will be constructed.

    If boolean flags with the label ``PARTIAL`` or ``CENTRIC`` are found
    in dataset, these will be cast to the ``MTZInt`` dtype, and included
    in the output MTZ file.

    Parameters
    ----------
    dataset : DataSet
        DataSet object to be written to MTZ file
    mtzfile : str or file
        name of an mtz file or a file object
    skip_problem_mtztypes : bool
        Whether to skip columns in DataSet that do not have specified
        MTZ datatypes
    project_name : str
        Project name to assign to MTZ file
    crystal_name : str
        Crystal name to assign to MTZ file
    dataset_name : str
        Dataset name to assign to MTZ file
    """
    mtz = to_gemmi(
        dataset, skip_problem_mtztypes, project_name, crystal_name, dataset_name
    )
    mtz.write_to_file(mtzfile)
    return


def read_cif(ciffile):
    """
    Populate the dataset object with reflection data from a CIF/ENT file.

    Reflections from CIF/ENT files are merged.
    A merged reflection DataSet will always be constructed.

    The function reads a CIF/ENT file and returns a rs.DataSet.

    Parameters
    ----------
    ciffile : str or file
        name of a CIF reflection file or a file object

    Returns
    -------
    DataSet
    """
    gemmi_cif = gemmi.cif.read(ciffile)
    rblocks = gemmi.as_refln_blocks(gemmi_cif)
    rblock = rblocks[0]
    gemmi_cif = gemmi.CifToMtz().convert_block_to_mtz(rblock)
    return from_gemmi(gemmi_cif)
