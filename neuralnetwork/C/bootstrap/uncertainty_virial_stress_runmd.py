"""This script is to test running MD simulation to compute the virial test for model
ensembles obtained from the DROPOUT.
model.
"""

from pathlib import Path
import json
import re
import jinja2
import os
import sys
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
from lammps import lammps
from ase.units import create_units

# Read setting file
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
DATA_DIR = ROOT_DIR / "data"
with open(ROOT_DIR / "settings.json", "r") as f:
    settings = json.load(f)
partition = settings["partition"]
suffix = "_".join([str(n) for n in settings["Nnodes"]])
PART_DIR = DATA_DIR / f"{partition}_partition_data"
RES_DIR = WORK_DIR / "results" / f"{partition}_partition_{suffix}"

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
lammps_template = """# Initialize interatomic potential (KIM model) and units
atom_style	atomic
kim             init {{ modelname }} metal
# boundary conditions
boundary	p p p
# create a honeycomb lattice
variable	a equal {{ latparam }}
variable	sqrt3 equal sqrt(3.0)
variable	b equal $a/${sqrt3}
variable	sc equal 3.0*$b  # Lattice scale
variable	hsc equal 34/${sc}  # Scaled van der Waals thickness
variable	asc equal $a/${sc}  # Scaled a
lattice		custom ${sc} &
			a1 1.0 0.0 0.0 &
			a2 0.0 ${asc} 0.0 &
			a3 0.0 0.0 ${hsc} &
			basis 0.0 0.0 0.0 &
			basis $(1.0/3.0) 0.0 0.0 &
			basis 0.5 0.5 0.0 &
			basis $(5.0/6.0) 0.5 0.0
# create simulation box and atoms
region		box block 0 4 0 6 -0.5 0.5 units lattice
create_box	1 box
create_atoms	1 box
mass		1 12

# specify atom type to chemical species mapping for the KIM model
kim		interactions C

# Equilibrate for 10_000 time steps
variable	temp equal 300.0
velocity        all create ${temp} 2023 mom yes rot yes
fix             mom all momentum 1 linear 1 1 1 angular
fix		mynve all nve
fix		mylgv all langevin 0.0 ${temp} 0.01 2023


thermo		100
thermo_style	custom step temp etotal press lx ly lz
run		10000
unfix           mylgv

# Start recording the data
compute         virial all pressure NULL virial
dump		mydump all custom 10 "{{ path }}/virial.dump" id type x y z fx fy fz
dump_modify	mydump format float %.16f
# Using this dump format, I can use ase.io.read to read the file and get all the
# information I need, including the atomic positions and forces.

# MD
fix		mylgv all langevin ${temp} ${temp} 0.1 2023
reset_timestep	0
thermo		10
thermo_style	custom step temp etotal press lx ly lz c_virial[*]
run		10000

undump		mydump
"""


def write_lammps_command(modelname, a, dumppath):
    """Use jinja2 templating to write commands to run MD simulation in lammps for virial
    stress calculation.

    Parameters
    ----------
    modelname: str
        KIM ID of the model. Make sure that the model is installed.
    a: float
        Lattice paramemter.
    dumppath: str or Path
        Where to store the trajector files dumped by lammps. These files contain
        information to compute the virial stress tensor.

    Returns
    -------
    lammps_command: str
        Rendered commands for running MD in lammps.
    """
    # Render lammps command
    env = jinja2.Environment()
    template = env.from_string(lammps_template)
    lammps_command = template.render(modelname=modelname, latparam=a, path=dumppath)
    return lammps_command


def run_one_latparam(modelname, a, dumppath, cmdargs=["-screen", os.devnull, "-nocite"]):
    """Run MD simulation for virial stress calculation for a given single lattice
    parameter. For the arguments of this function, they are mostly the same as the
    arguments for ``write_lammps_command``.

    Parameters
    ----------
    cmdargs: list
        ``cmdargs`` argument in ``lammps.lammps``.
    """
    # Prepare the dump folder
    dumppath = Path(dumppath)
    if not dumppath.exists():
        dumppath.mkdir(parents=True)
    # Write lammps command string
    lammps_command = write_lammps_command(modelname, a, dumppath)
    # Run lammps
    lmp = lammps(cmdargs=cmdargs)
    lmp.commands_string(lammps_command)
    lmp.close()


def post_process_outfile(outfile):
    """Read the output file of lammps and retrieve the stress tensor mean.
    Note that the stress values are in bars. To convert to GPa, do stress * u["bar"] / u["GPa"]
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
            values = np.row_stack((values, numbers))

            if int(numbers[0]) == 1000:
                break
    stress_ens = values[:, -6:]
    # Fix the volume
    # The volume used in MD is 10 times the van der Waals thickness
    stress_ens *= 10
    return np.mean(stress_ens, axis=0)


def run_md_one_latparam(a, modelname, savepath):
    """Run the MD simulation and get the virial stress for a given lattice parameter and
    index ensemble.
    """
    # print("Lattice parameter:", a)

    path = Path(savepath) / f"a{a:0.3f}"
    if not path.exists():
        path.mkdir(parents=True)

    outfile = path / f"lammps_a{a:0.3f}.out"
    if not outfile.exists():
        run_one_latparam(modelname, a, path, cmdargs=["-screen", str(outfile)])
    # Post-process
    return post_process_outfile(outfile)


if __name__ == "__main__":
    argv = sys.argv

    alist = np.linspace(0.95, 1.05, 11) * 2.466
    idx = int(argv[1]) if len(argv) > 1 else 0
    modelname = f"DUNN_C_bootstrap_{idx:03d}"
    path = RES_DIR / f"{idx:03d}" / "virial_stress"

    def run_md_one_latparam_wrapper(a):
        return run_md_one_latparam(a, modelname, path)

    with Pool(len(alist)) as p:
        stress = np.array(
            list(tqdm(p.imap(run_md_one_latparam_wrapper, alist), total=len(alist)))
        )
