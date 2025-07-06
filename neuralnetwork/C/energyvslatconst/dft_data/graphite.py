from pathlib import Path
import re
import pickle

import numpy as np
from ase.lattice.hexagonal import Graphite
from ase.calculators.vasp import Vasp
from ase.io import read, write
from scipy.optimize import minimize, least_squares

import matplotlib.pyplot as plt


##########################################################################################
# Setup
# -----

# Setting up directory
FILE_DIR = Path(__file__).resolve().parent
XYZ_DIR = FILE_DIR / "xyz_files" / "graphite"

# Isolated atom energy
isolated_energy = -1.24286938  # This is the value that Mingjian sent me
print(f"Isolated carbon atom energy: {isolated_energy:.6f} eV")

# Vasp calculator
calc = Vasp(
    # Command to run VASP
    command="mpirun -np 16 vasp_std",
    directory="graphite",
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
    ncore=4,
)
calc.clean()


##########################################################################################
# Energy vs lattice parameter
# ---------------------------
print("Energy vs lattice parameter")

# Lattice parameters - Extended
a0 = 2.466
c0 = 6.7
alist_scaled = np.linspace(0.93, 1.09, 21)
diff = 0.008
# Left
alist_scaled = np.append(np.arange(-15, 0) * diff + alist_scaled[0], alist_scaled)
# Right
alist_scaled = np.append(alist_scaled, np.arange(1, 15) * diff + alist_scaled[-1])
alist = alist_scaled * a0


energy_file = FILE_DIR / "graphite_new.txt"
if energy_file.exists():
    alist, clist, energy = np.loadtxt(energy_file, delimiter=",").T
else:
    # Iterate over lattice parameters
    clist = np.zeros_like(alist)
    energy = np.zeros_like(alist)
    for ii, a in enumerate(alist):
        print("Lattice parameter:", a)

        # Issue: For graphite, given a lattice parameter a, we will relax the lattice
        # parameter c. The following function computes the energy for a fix a given c.
        # This function can be used to find the optimal c.
        def energy_given_c(c_array):
            # Create the lattice
            c = c_array[0]
            print("Evaluate energy with lattice parameter (a, c):", [a, c])
            atoms = Graphite("C", latticeconstant={"a": a, "c": c})
            # Assign calculator
            atoms.calc = calc
            # Energy calculation
            eng = atoms.get_potential_energy()
            calc.clean()  # Clean up Vasp files
            return eng

        # Optimize c for a given a
        opt = minimize(
            energy_given_c,
            [c0],
            method="cg",
            jac="3-point",
            tol=1e-5,
            options={"eps": 1e-2, "disp": True},
        )
        # Save raw optimization result
        with open(XYZ_DIR / f"graphite_{a:.3f}.pkl", "wb") as f:
            pickle.dump(opt, f)

        # Create the atoms object
        atoms = Graphite("C", latticeconstant={"a": a, "c": opt.x[0]})
        natoms = atoms.get_global_number_of_atoms()
        # We can get the energy value from the optimization result
        eng = opt.fun

        # Post-process
        # Lattice parameter c
        clist[ii] = opt.x[0]
        # Energy per atom
        eng /= natoms
        # Subtract the isolated atom energy
        eng -= isolated_energy
        # Write xyz file with energy data that can be read by KLIFF
        conf_eng = eng + natoms * eng
        atoms.write(XYZ_DIR / f"graphite_{a:.3f}.xyz", format="extxyz")
        # Finalize --- collect
        energy[ii] = eng
        print("Energy:", energy[ii])

        # Save the data
        alist_energy = np.column_stack((alist, clist, energy))
        np.savetxt(energy_file, alist_energy, delimiter=",")

# ##########################################################################################
# # Visualization
# # -------------
# # Lattice parameter c
# plt.figure()
# plt.plot(alist, clist, "o-")
# plt.ylabel(r"Lattice parameter c ($\AA$)")
# plt.xlabel(r"Lattice parameter a ($\AA$)")
# plt.show()

# # Energy
# plt.figure()
# plt.plot(alist, energy, "o-")
# plt.ylabel("Energy (eV)")
# plt.xlabel(r"Lattice parameter ($\AA$)")
# plt.show()
