from pathlib import Path
import re

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.vasp import Vasp
from ase.io import read, write

import matplotlib.pyplot as plt


##########################################################################################
# Setup
# -----

# Setting up directory
FILE_DIR = Path(__file__).resolve().parent
XYZ_DIR = FILE_DIR / "xyz_files" / "diamond"

# Isolated atom energy
isolated_energy = -1.24286938  # This is the value that Mingjian sent me
print(f"Isolated carbon atom energy: {isolated_energy:.6f} eV")

# Vasp calculator
calc = Vasp(
    # Command to run VASP
    command="mpirun -np 4 vasp_std",
    directory="diamond",
    # Energy cutoff
    encut=500,
    # Exchange-correlation functional
    xc="PBE",
    # Many-body dispersion correction
    ivdw=21,
    lvdw_ewald=True,
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
)
calc.clean()


##########################################################################################
# Energy vs lattice parameter
# ---------------------------
print("Energy vs lattice parameter")

# Lattice parameters
a0 = 3.56  # Equilibrium lattice parameter for diamond
alist = np.linspace(0.93, 1.09, 21) * a0  # List of lattice parameters to calculate

energy_file = FILE_DIR / "diamond.txt"
if energy_file.exists():
    alist, energy = np.loadtxt(energy_file, delimiter=",").T
else:
    # Iterate over lattice parameters
    energy = np.zeros_like(alist)
    for ii, a in enumerate(alist):
        print("Lattice parameter:", a)
        # Create the atoms object
        atoms = bulk("C", "diamond", a=a)
        natoms = atoms.get_global_number_of_atoms()
        atoms.calc = calc
        # Let's export the atoms object to a file
        # Energy calculation
        eng = atoms.get_potential_energy()
        calc.clean()  # Clean up Vasp files

        # Post-process
        # Energy per atom
        eng /= natoms
        # Subtract the isolated atom energy
        eng -= isolated_energy
        # Write xyz file with energy data that can be read by KLIFF
        conf_eng = eng + natoms * eng
        atoms.write(XYZ_DIR / f"diamond_{a:.3f}.xyz", format="extxyz")
        # Finalize --- collect
        energy[ii] = eng
        print("Energy:", energy[ii])

    # Save the data
    alist_energy = np.column_stack((alist, energy))
    np.savetxt(energy_file, alist_energy, delimiter=",")

# ##########################################################################################
# # Visualization
# # -------------
# plt.figure()
# plt.plot(alist, energy, "o-")
# plt.ylabel("Energy (eV)")
# plt.xlabel(r"Lattice parameter ($\AA$)")
# plt.show()
