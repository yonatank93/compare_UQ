"""This script will be used to generate graphite and graphene sheet. Note that there is a
specific convention that we need to follow to specify the lattice vectors and atom
positions in LAMMPS.
* When defining the lattice, LAMMPS has a required `scale` argument. The lattice vectors
  should be written with this scaling factor.
* The lattice vectors specified as `a1`, `a2`, and `a3` CAN have elements >1.0.
* The atom positions needs to be written in basis of lattice vectors. Note that no matter
  how many cell we will use in the simulation, we only need to specify the positions in
  the unit cell.
"""

import copy

import numpy as np


def generate_unit_cell_graphene(a, unit="lattice", zshift=0.0):
    """Generate the atomic position and lattice vectors of 1 unit cell of graphene, given
    the vertical lattice constant :math:`a`. To generate a larger sheet of graphene, the
    unit cell needs to be translated and copied in the x direction by :math:`a \sqrt{3}`
    and in the y direction by :math:`a`.

    Parameters
    ----------
    a: float
        Lattice constant of graphene to generate. The values here would be written in
        unit angstrom.
    unit: str (optional)
        This determine the values of the atomic position to return. If "lattice", then
        the atomic positions will be written as a fraction of the lattice vectors, which
        means that the coordinates would be less than 1.0. This will be useful to use in
        lammps script. If "angstrom", then the positions returned will be in angstrom.
        This option will be useful to write xyz file.
    zshift: float (optional)
        How much to shift in the z direction. For example, if the sheet is to put in the
        middle of unit cell in z direction, then we need to shift the atomic positions
        in z direction up. This might be desireable so that the atoms wont jump across
        periodic boundary in z direction, especially since we make the cell huge in z
        direction. Please input the number based on the unit used.

    Returns
    -------
    cell: np.ndarray (3, 3,)
        Lattice vectors. Each vector is given as a column vector.
    pos: np.ndarray (4, 3,)
        Positions of the atoms in a graphene unit cell.
    """
    # Lattice vectors
    v1 = np.array([np.sqrt(3), 0, 0])  # Aligned with x axis
    v2 = np.array([0, 1, 0])  # Aligned with y axis
    # Make sure the atoms are not interacting across periodic boundary in z direction
    v3 = np.array([0, 0, 1])  # ALigned with z axis
    cell = np.column_stack((v1, v2, v3))

    # Atomic positions
    # Lattice unit
    a1 = np.array([0, 0, zshift])
    a2 = np.array([1 / 3, 0, zshift])
    a3 = np.array([0.5, 0.5, zshift])
    a4 = np.array([5 / 6, 0.5, zshift])
    pos = np.row_stack((a1, a2, a3, a4))

    # Angstrom unit
    if unit.lower() == "angstrom":
        b = a / np.sqrt(3)
        pos[:, 0] *= 3 * b
        pos[:, 1] *= a
        pos[:, 2] *= 30
        cell *= a

    return cell, pos


def generate_graphene_sheet(a, nx, ny, unit="lattice", zshift=0.0):
    """Generate a sheet of graphene. Many of the arguments will be passed in to
    `generate_unit_cell_graphene` function, particularly `a`, `unit`, and `zshift`.

    Parameters
    ----------
    a: float
        Lattice constant of graphene to generate. The values here would be written in
        unit angstrom.
    nx: int
        How many unit cells in the x direction.
    ny: int
        How many unit cells in the y direction.
    unit: str (optional)
        This determine the values of the atomic position to return.
    zshift: float (optional)
        How much to shift in the z direction.

    Returns
    -------
    cell: np.ndarray (3, 3,)
        Lattice vectors. Each vector is given as a column vector.
    pos: np.ndarray (4 * nx * ny, 3,)
        Positions of the atoms in a graphene unit cell.

    Notes
    -----
    I don't think we will need this fnction in generating lammps script, since lammps only
    need the unit cell information. However, this function might be useful if we want to
    plot the structure, for example for illustration in the presentation etc.
    """
    # Cell and atomic positions in a unit cell, in unit lattice with zero z shift
    cell0, pos0 = generate_unit_cell_graphene(a)

    # Lattice vectors
    cell = copy.deepcopy(cell0)
    cell[0] *= nx
    cell[1] *= ny

    # Copy the unit cell to generate the entire sheet
    pos = np.empty((0, 3))
    for y in range(ny):
        for x in range(nx):
            pos0_shifted = copy.deepcopy(pos0)
            pos0_shifted[:, 0] += x
            pos0_shifted[:, 1] += y
            pos = np.row_stack((pos, pos0_shifted))

    # Scale the position to be between 0 and 1
    pos[:, 0] /= nx
    pos[:, 1] /= ny
    # Shift in z direction
    pos[:, 2] += zshift

    # Angstrom
    if unit.lower() == "angstrom":
        b = a / np.sqrt(3)
        pos[:, 0] *= 3 * b * nx
        pos[:, 1] *= a * ny
        pos[:, 2] *= 30
        cell *= a

    return cell, pos


def generate_unit_cell_graphite(a, c, unit="lattice"):
    """Generate the atomic position and lattice vectors of 1 unit cell of graphene, given
    the vertical lattice constant :math:`a`. To generate a larger sheet of graphene, the
    unit cell needs to be translated and copied in the x direction by :math:`a \sqrt{3}`
    and in the y direction by :math:`a`.

    Parameters
    ----------
    a: float
        Lattice constant of graphene to generate. The values here would be written in
        unit angstrom.
    unit: str (optional)
        This determine the values of the atomic position to return. If "lattice", then
        the atomic positions will be written as a fraction of the lattice vectors, which
        means that the coordinates would be less than 1.0. This will be useful to use in
        lammps script. If "angstrom", then the positions returned will be in angstrom.
        This option will be useful to write xyz file.

    Returns
    -------
    cell: np.ndarray (3, 3,)
        Lattice vectors. Each vector is given as a column vector.
    pos: np.ndarray (4, 3,)
        Positions of the atoms in a graphene unit cell.
    """
    # Lattice vectors
    v1 = np.array([np.sqrt(3), 0, 0])  # Aligned with x axis
    v2 = np.array([0, 1, 0])  # Aligned with y axis
    v3 = np.array([0, 0, c / a])  # ALigned with z axis
    cell = np.column_stack((v1, v2, v3))

    # Atomic positions
    # Lattice unit
    a1 = np.array([0, 0, 0])
    a2 = np.array([1 / 3, 0, 0])
    a3 = np.array([0.5, 0.5, 0])
    a4 = np.array([5 / 6, 0.5, 0])
    a5 = np.array([1 / 6, 0.5, 0.5])
    a6 = np.array([1 / 3, 0, 0.5])
    a7 = np.array([2 / 3, 0, 0.5])
    a8 = np.array([5 / 6, 0.5, 0.5])
    pos = np.vstack((a1, a2, a3, a4, a5, a6, a7, a8))

    # Angstrom unit
    if unit.lower() == "angstrom":
        b = a / np.sqrt(3)
        pos[:, 0] *= 3 * b
        pos[:, 1] *= a
        pos[:, 2] *= c
        cell *= a

    return cell, pos
