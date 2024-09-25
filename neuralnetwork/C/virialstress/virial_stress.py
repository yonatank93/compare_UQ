"""This script is to test running MD simulation to compute the virial test for model
ensembles obtained from the DROPOUT.
model.
"""

from pathlib import Path
import re
import jinja2
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

import numpy as np
from lammps import lammps
from ase.units import create_units

# Read settings
WORK_DIR = Path(__file__).absolute().parent

u = create_units("2018")

# Regex to find numbers
numeric_const_pattern = r"""
    [-+]? # optional sign
    (?:
        (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
        |
        (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
    )
    # followed by optional exponent part if desired
    (?: [Ee] [+-]? \d+ ) ?
    """
rx = re.compile(numeric_const_pattern, re.VERBOSE)

# Lammps script template
template_file = WORK_DIR / "graphene.tpl"


def write_lammps_command(modelname, a, idx, dumppath):
    """Use jinja2 templating to write commands to run MD simulation in lammps for virial
    stress calculation.

    Parameters
    ----------
    modelname: str
        KIM ID of the model. Make sure that the model is installed.
    a: float
        Lattice paramemter.
    idx: int
        Active member id of the model. This is mainly important when using dropout
        ensemble.
    dumppath: str or Path
        Where to store the trajector files dumped by lammps. These files contain
        information to compute the virial stress tensor.

    Returns
    -------
    lammps_command: str
        Rendered commands for running MD in lammps.
    """
    # Render lammps command
    loader = jinja2.FileSystemLoader(WORK_DIR)
    environment = jinja2.Environment(loader=loader)
    template = environment.get_template(template_file.name)
    lammps_command = template.render(
        modelname=modelname, latparam=a, idx=idx, dumppath=dumppath
    )
    return lammps_command


def run_lammps(modelname, a, idx, dumppath, cmdargs=["-screen", os.devnull, "-nocite"]):
    """Run MD simulation for virial stress calculation for a given single lattice
    parameter using LAMMPS. For the arguments of this function, they are mostly the same
    as the arguments for ``write_lammps_command``.

    Parameters
    ----------
    modelname: str
        KIM ID of the model. Make sure that the model is installed.
    a: float
        Lattice paramemter.
    idx: int
        Active member id of the model. This is mainly important when using dropout
        ensemble.
    dumppath: str or Path
        Where to store the trajector files dumped by lammps. These files contain
        information to compute the virial stress tensor.
    cmdargs: list
        ``cmdargs`` argument in ``lammps.lammps``.
    """
    # print("Run LAMMPS", a)
    # Prepare the dump folder
    dumppath = Path(dumppath)
    dumppath.mkdir(parents=True, exist_ok=True)
    # Take care of the active member index
    if idx is None:
        kim_param_str = ""
    else:
        idx += 1
        kim_param_str = f"kim param set active_member_id 1 {idx}"
    # Write lammps command string
    lammps_command = write_lammps_command(modelname, a, kim_param_str, dumppath)
    # Run lammps
    lmp = lammps(cmdargs=cmdargs)
    lmp.commands_string(lammps_command)
    lmp.close()


def post_process_outfile(outfile):
    """Read the output file of lammps and retrieve the stress tensor mean.
    Note that the stress values are in bars. To convert to GPa, do stress * u["bar"] / u["GPa"]

    Parameters
    ----------
    outfile: str or Path-like
        Path to the output file that the calculation produce. This output file contains
        the positions and forces data that we use to compute the virial stress.
    """
    # Post-process
    # Read lammps out file
    with open(outfile, "r") as f:
        out_lines = f.readlines()

    # Get the information from the file
    record = False
    values = np.zeros((0, 13))
    for line in out_lines:
        if "c_virial" in line:
            record = True
            continue

        if record:
            numbers = [float(val) for val in rx.findall(line)]
            values = np.vstack((values, numbers))

            if int(numbers[0]) == 1000:
                break
    stress_ens = values[:, -6:]
    # Fix the volume
    # The volume used in MD is 10 times the van der Waals thickness
    stress_ens *= 10
    return np.mean(stress_ens, axis=0)


def run_md_one_latparam(a, modelname, idx, savepath):
    """Run the MD simulation and get the virial stress for a given lattice parameter and
    index ensemble.

    Parameters
    ----------
    a: float
        Lattice paramemter.
    modelname: str
        KIM ID of the model. Make sure that the model is installed.
    idx: int
        Active member id of the model. This is mainly important when using dropout
        ensemble.
    savepath: str or Path
        Where to store the trajector files dumped by lammps. These files contain
        information to compute the virial stress tensor.
    """
    # print("Lattice parameter:", a)

    path = Path(savepath) / f"a{a:0.3f}"
    path.mkdir(parents=True, exist_ok=True)

    outfile = path / f"lammps_a{a:0.3f}.out"
    if not outfile.exists():
        run_lammps(modelname, a, idx, path, cmdargs=["-screen", str(outfile)])
    # Post-process
    return post_process_outfile(outfile)


def virialvslatconst(modelname, alist, idx=None, savepath=Path()):
    """Iterate over the list of lattice parameters and compute the virial stress for each
    lattice parameter value.

    Parameters
    ----------
    modelname: str
        KIM ID of the model. Make sure that the model is installed.
    alist: list or np.ndarray
        A list of lattice parameters.
    idx: int
        Active member id of the model. This is mainly important when using dropout
        ensemble. If None, then use the model default.
    savepath: str or Path
        Where to store the trajector files dumped by lammps. These files contain
        information to compute the virial stress tensor.
    """
    run_md_one_latparam_wrapper = partial(
        run_md_one_latparam, modelname=modelname, idx=idx, savepath=savepath
    )

    with Pool(len(alist)) as p:
        stress = np.array(
            list(tqdm(p.imap(run_md_one_latparam_wrapper, alist), total=len(alist)))
        )

    return stress
