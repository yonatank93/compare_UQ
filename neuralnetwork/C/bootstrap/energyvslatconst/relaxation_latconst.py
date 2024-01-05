from pathlib import Path
import jinja2
import os

import numpy as np
import lammps

try:
    from .generate_graphene import generate_unit_cell
except ImportError:
    from generate_graphene import generate_unit_cell

WORK_DIR = Path(__file__).absolute().parent
template_file = "relaxation.tpl"
outfile = WORK_DIR / "relaxation.in"


def equilibrate_graphene(potential, ainit, nx=1, ny=1, debug=False):
    """Relax a graphene structure and get the equilibrium lattice constant and cohesive
    energy.

    Parameters
    ----------
    potential: str
        KIM ID for the potential to use.
    ainit: float
        Initial guess of lattice constant in angstrom.
    nx, ny: int (optional)
        Number of cell in x and y direction, respectively.
    debug: bool (optional)
        If True, then the generated lammps script will be returned as well.

    Returns
    -------
    a0: float
        Equilibrium lattice constant a in angstrom.
    e0: float
        Cohesive energy in eV.
    """
    # Generate graphene sheet
    cell, pos = generate_unit_cell(ainit, unit="lattice", zshift=0.0)
    natoms = len(pos)

    # Convert the cell and position information above to lammps string
    # Cell "a1 a1x a1y a1z a2 a2x a2y a2z a3 a3x a3y a3z"
    cell_str = (
        "    "
        + f"a1 {cell[0, 0]} {cell[1, 0]} {cell[2, 0]} "
        + f"a2 {cell[0, 1]} {cell[1, 1]} {cell[2, 1]} "
        + f"a3 {cell[0, 2]} {cell[1, 2]} {cell[2, 2]}"
    )
    # Positions "basis b1x b1y b1z basis b2x b2y b2z ..."
    pos_str = ""
    for ii, atom in enumerate(pos):
        pos_str_peratom = "    " + f"basis {atom[0]} {atom[1]} {atom[2]} &\n"
        if ii == natoms - 1:
            pos_str_peratom = pos_str_peratom[:-3]
        pos_str += pos_str_peratom

    # Write input file
    loader = jinja2.FileSystemLoader(WORK_DIR)
    environment = jinja2.Environment(loader=loader)
    template = environment.get_template(template_file)
    content = template.render(
        potential=potential, inita=ainit, cell=cell_str, pos=pos_str, nx=nx, ny=ny
    )
    # with open(outfile, "w") as f:
    #     f.write(content)
    # print(content)

    # Run lammps script
    lmp = lammps.lammps(cmdargs=["-screen", os.devnull, "-nocite"])
    lmp.commands_string(content)
    a0 = lmp.extract_variable("length")  # Equilibrium lattice constant
    e0 = lmp.extract_variable("ecoh")  # Equilibrium cohesive energy
    lmp.close()

    if debug:
        return a0, e0, content
    else:
        return a0, e0


if __name__ == "__main__":
    from datetime import datetime

    potential = "DUNN_WenTadmor_2019v1_C__MO_584345505904_000"
    ainit = 2.47  # Lattice constant from  materials project for graphite-like structure
    start = datetime.now()
    a0, e0 = equilibrate_graphene(potential, ainit, 1, 1)
    end = datetime.now()
    print("Evaluation time:", end - start)
