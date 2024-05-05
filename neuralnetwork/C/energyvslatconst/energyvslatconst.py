"""This script contains calculation of energy as a function of lattice constant for
graphene structure.
"""

from pathlib import Path
import jinja2
import os

import numpy as np
import lammps

try:
    from .generate_graphene import generate_unit_cell
    from .relaxation_latconst import equilibrate_graphene
except ImportError:
    from generate_graphene import generate_unit_cell
    from relaxation_latconst import equilibrate_graphene


WORK_DIR = Path(__file__).absolute().parent
# template_file = "energy_latconst.tpl"
avail_struct = ["graphene", "diamond"]


def energyvslatconst(
    potential, alist, structure="graphene", active_member_id=0, lmpfile=None
):
    """Compute the energy vs lattice constant curve for given a list of lattice constant
    a. Additionally, a list of lattice constant b from the relaxation will be returned.
    The relaxation is done in LAMMPS.

    Parameters
    ----------
    potential: str
        KIM ID for the potential to use.
    alist: np.ndarray
        A list of lattice constant, preferably in ascending order.
    structure: str
        Structure of the carbon material. It can only be "diamond" and "graphene".
    active_member_id: int
        Integer number that sets the active member in the ensemble. 0 means to not use
        dropout, -1 means to take the mean across all ensemble, and 1-100 correspond to
        each dropout ensemble member.

    Returns
    -------
    elist: np.ndarray
        A list of energy values for given lattice constants.
    eng_error: np.ndarray
        Standard deviations of the energy, calculated using dropout ensemble.
    """
    # Get the correct lammps template
    if structure == "graphene":
        template_file = "energy_latconst.tpl"
    elif structure == "diamond":
        template_file = "energy_latconst_diamond.tpl"
    else:
        raise ValueError(f"Unknown structure, available structure: {avail_struct}")

    # Vary lattice constants
    predictions = np.empty((0, 3))

    for a in alist:
        preds_a = energy_given_latconst(
            potential, a, active_member_id, template_file, lmpfile
        )
        # Append the result
        predictions = np.row_stack((predictions, preds_a))

    return predictions.T


def energy_given_latconst(potential, a, active_member_id, template_file, lmpfile=None):
    loader = jinja2.FileSystemLoader(WORK_DIR)
    environment = jinja2.Environment(loader=loader)
    tempfile = template_file
    template = environment.get_template(tempfile)
    content = template.render(potential=potential, a=a, set_id=active_member_id)

    # Run lammps script
    lmp = lammps.lammps(cmdargs=["-screen", os.devnull, "-nocite"])
    if lmpfile is None:
        lmp.commands_string(content)
    else:
        with open(lmpfile, "w") as f:
            f.write(content)
        lmp.file(lmpfile)
    eng_mean = lmp.extract_variable("E_mean")
    eng_error = lmp.extract_variable("E_stdev")
    lmp.close()
    return a, eng_mean, eng_error


if __name__ == "__main__":
    from datetime import datetime
    import matplotlib.pyplot as plt

    # Equilibration
    # potential = "DUNN_WenTadmor_2019v1_C__MO_584345505904_000"
    potential = "DUNN_best_train"
    ainit = 2.466  # From materials project for graphite-like structure
    a0, e0 = equilibrate_graphene(potential, ainit)
    alist = np.linspace(0.93, 1.07, 16) * a0

    start = datetime.now()
    alist, elist, errlist = energyvslatconst(potential, alist)
    end = datetime.now()
    print("Evaluation time:", end - start)

    # Load DFT data
    dft_data = np.loadtxt("dft_data.txt", delimiter=",")

    # Plot the result curves
    # Energy vs lattice constant
    plt.figure()
    plt.plot(*(dft_data.T), "r.", zorder=10, label="DFT")
    plt.plot(alist, elist, "-", label="DUNN mean")
    plt.fill_between(
        alist,
        elist - errlist,
        elist + errlist,
        alpha=0.5,
        zorder=-1,
        label="DUNN uncertainty",
    )
    plt.ylim(-8.1, -7.5)
    plt.xlabel(r"Lattice constant $a$ $(\AA)$")
    plt.ylabel("Energy (eV/atom)")
    plt.legend()

    plt.show()
