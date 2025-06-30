"""Generate perturbed supercells for force constant calculation, then for each
perturbation submit force calculation using VASP calculator.
"""

from pathlib import Path
import pickle
import numpy as np
from jinja2 import Template
import subprocess
import json

from ase.lattice.hexagonal import Graphite
from ase.vibrations.vibrations import Displacement as VDisplacement
from ase.io import read

FILE_DIR = Path(__file__).parent
NCELL = 5
TARGET_DIR = FILE_DIR / f"phonon_graphite_{NCELL}x{NCELL}x2"


#########################################################################################
# CREATE DISPLACED SUPERCELLS
# ===========================
print("Creating displaced supercells...")

# Unit cell
atoms = Graphite("C", latticeconstant={"a": 2.46, "c": 6.7})

# Create displaced supercells that mimic ASE phonon module
# ph = Phonons(atoms, calc, supercell=(NCELL, NCELL, 2), delta=0.01, name=savedir)
supercell = (NCELL, NCELL, 2)
delta = 0.1

indices = np.arange(len(atoms))
atoms_N = atoms * supercell
offset = 0
# Positions of atoms to be displaced in the reference cell
natoms = len(atoms)
offset = natoms * offset
pos = atoms_N.positions[offset : offset + natoms].copy()

# Loop over all displacements
subfolders = []
for a in indices:
    for i, cart in enumerate(["x", "y", "z"]):
        for sign in [-1, 1]:
            atoms_N.positions[offset + a, i] = pos[a, i] + sign * delta
            # Export
            sign_str = "+" if sign == 1 else "-"
            folder = TARGET_DIR / f"displaced_{a}{cart}{sign_str}"
            subfolders.append(folder)
            folder.mkdir(parents=True, exist_ok=True)
            atoms_N.write(folder / "atoms.xyz", format="extxyz")
            print("Saved displaced atoms to", folder.name)
            # Reset atoms position
            atoms_N.positions[offset + a, i] = pos[a, i]
# Don't forget to also export the unperturbed supercell
folder = TARGET_DIR / "displaced_eq"
folder.mkdir(parents=True, exist_ok=True)
subfolders.append(folder)
atoms_N.write(folder / "atoms.xyz", format="extxyz")
print("Saved displaced atoms to", folder.name)
print()


#########################################################################################
# WRITE SCRIPTS TO COMPUTE THE FORCES
# ===================================
print("Writing scripts to compute the forces...")

# Python script --- Name the rendered file as forces.py
python_script = """
from pathlib import Path
import json

from ase.io import read
from ase.calculators.vasp import Vasp

FILE_DIR = Path(__file__).parent

# Read the atoms
atoms = read(FILE_DIR / "atoms.xyz", format="extxyz")
# Vasp calculator
calc = Vasp(
    # Command to run VASP
    command="mpirun -np 128 vasp_std",
    directory=FILE_DIR,
    # Energy cutoff
    encut=500,
    # Exchange-correlation functional
    xc="PBE",
    # Many-body dispersion correction
    ivdw=202,
    # lvdw_ewald=True,
    # Electronic relaxation
    prec="accurate",  # use "accurate" to get highly accureate forces
    nelm=60,  # maximum number of electronic SC (selfconsistency) steps; Default: 60
    ediff=1e-6,  # Default:1E-04    we need higher precision converging to meV
    ismear=0,  # Smearing method to partially fill orbits; 0 is Gaussian smearing
    sigma=0.05,  # Small smearing width
    # k-points
    kpts=[8, 8, 8],
    gamma=True,  # Centered on Gamma point
    # Write some file or not
    lwave=False,
    lcharg=False,
    # For parallelization
    ncore=32,
)
calc.clean()
# Assign calculator to the atoms object
atoms.calc = calc

# Compute forces
atoms.get_forces()

# Export as .traj binary file so that we won't loose any precision
atoms.write(FILE_DIR / "forces.traj")
"""


# SLURM template --- Name the rendered file as submit.sh
slurm_template = """#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=165:00:00   # walltime
#SBATCH --ntasks=128   # number of processor cores
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH --job-name={{ job_name }}   # job name
#SBATCH --mail-user=kurniawanyo@outlook.com   # email address
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source ~/.bash_profile

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn.
# Does nothing if the program doesn't use OpenMP.
# export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUM_CORES=${SLURM_NTASKS}
echo "Running ${SLURM_JOB_NAME}"
echo "Running on ${NUM_CORES} cores"

ulimit -s unlimited

time python {{ python_file }}

echo "All Done!" 
"""

# Iterate
for folder in subfolders:
    print("Write forces.py and submit.sh to", folder.name)
    suffix = folder.name.split("_")[1]

    # Write the python script
    with open(folder / "forces.py", "w") as f:
        f.write(python_script)

    # Write the slurm script
    template = Template(slurm_template)
    slurm_script = template.render(
        job_name=f"phonon_graphite_{NCELL}x{NCELL}x2_{suffix}",
        python_file=folder / "forces.py",
    )
    with open(folder / "submit.sh", "w") as f:
        f.write(slurm_script)
print()

#########################################################################################
# SUBMIT FORCE CALCULATION JOBS
# =============================
print("Submitting force calculation jobs...")


def save_forces_to_json(forces, filename):
    """This function will mimic the JSON cache writer in ASE."""
    shape = forces.shape
    dtype = forces.dtype.name
    save_dict = {"forces": {"__ndarray__": [shape, dtype, forces.flatten().tolist()]}}
    with open(filename, "w") as f:
        json.dump(save_dict, f)


# Iterate
for folder in subfolders:
    suffix = folder.name.split("_")[1]
    check_file = folder / f"forces.traj"
    if not check_file.exists():
        subprocess.run(f"sbatch {folder}/submit.sh", shell=True)
        print("Submitted job for", folder.name)
    else:
        print("Force calculation is already done for", folder.name)
        atoms = read(check_file)
        filename = folder / f"cache.{suffix}.json"
        save_forces_to_json(atoms.get_forces(), filename)
        # Copy the json file to the target directory so that I can use it with ASE phonon
        # module
        subprocess.run(f"cp {filename} {TARGET_DIR}", shell=True)
