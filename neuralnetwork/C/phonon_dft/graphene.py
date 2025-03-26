"""Use this script to generate the reference DFT data of the phonon dispersion curve for
graphene structure. The phonon calculation is done in ASE using VASP calculator.
"""

from pathlib import Path
import pickle

from ase.lattice.hexagonal import Graphene
from ase.calculators.vasp import Vasp
from ase.phonons import Phonons

import numpy as np
from scipy.optimize import least_squares

FILE_DIR = Path(__file__).parent
NCELL = 6
TARGET_DIR = FILE_DIR / f"phonon_graphene_{NCELL}x{NCELL}x1"

# Vasp calculator
calc = Vasp(
    # Command to run VASP
    command="mpirun -np 128 vasp_std",
    directory=TARGET_DIR,
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
    kpts=[16, 16, 1],
    gamma=True,  # Centered on Gamma point
    # Write some file or not
    lwave=False,
    lcharg=False,
    # For parallelization
    ncore=16,
)
calc.clean()


##########################################################################################
# Relaxation -- Find equilibrium lattice constant
# -----------------------------------------------
# print("Relaxation -- Find equilibrium lattice constant")
# # We will use the `least_squares` function from `scipy.optimize` to find the equilibrium
# # Since the energy is negative, we need to modify the function to optimize. Since I know
# # that the minimum energy is about -18 eV, I can just define a random baseline energy,
# # say -20 eV, and minimize the difference.
# baseline = -20  # Arbitrary baseline energy


# def energy_minimizer(a):
#     atoms = Graphene("C", latticeconstant={"a": a[0], "c": 30})
#     atoms.calc = calc
#     return np.array([atoms.get_potential_energy()]) - baseline


# # Optimization - Bound the lattice constant between 2.0 and 3.0
# opt = least_squares(energy_minimizer, [2.46], bounds=(2.0, 3.0))
# a0 = opt.x[0]

print("Relaxation is not needed, we will use DFT lattice constant")
a0 = 2.46

##########################################################################################
# Phonon calculation
# ------------------
print("Phonon calculation")
# Setup crystal and EMT calculator
atoms = Graphene("C", latticeconstant={"a": a0, "c": 30})
# Assign calculator tot he atoms object
atoms.calc = calc

# Phonon calculation
# Phonon calculator
ph = Phonons(atoms, calc, supercell=(NCELL, NCELL, 1), delta=0.01, name=TARGET_DIR)
ph.run()
calc.clean()  # Clean up files exported by the calculator
# Read forces and assemble the dynamical matrix
ph.read(acoustic=True)
# ph.clean()
# Get the band structure
path = atoms.cell.bandpath("GMKG", npoints=100)
bs = ph.get_band_structure(path)

# Export the results
energies = bs.energies
# Convert to THz
conversion = 4.136e-3  # 1 Thz = 4.136 meV
energies /= conversion

labels = list(bs.get_labels())
labels[2] = [r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"]

with open(TARGET_DIR / "phonon_graphene.pkl", "wb") as f:
    pickle.dump({"energies": energies, "labels": labels}, f, protocol=4)
