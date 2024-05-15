"""This script contains calculation of energy as a function of lattice constant for
graphene structure.
"""

from pathlib import Path
import jinja2
import os

import numpy as np
import lammps


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
    template_file = f"energy_latconst_{structure}.tpl"

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
