"""Use this script to generate the reference DFT data of the phonon dispersion curve for
graphite structure. The phonon calculation is done in ASE using VASP calculator.
"""

from pathlib import Path
import pickle
import argparse

from ase.lattice.hexagonal import Graphite
from ase.calculators.vasp import Vasp
from ase.phonons import Phonons

parser = argparse.ArgumentParser()
parser.add_argument("--correction", action="store_true")
args = parser.parse_args()

FILE_DIR = Path(__file__).parent
NCELL = 5
TARGET_DIR = FILE_DIR / f"phonon_graphite_{NCELL}x{NCELL}x2"

# Setup crystal and EMT calculator
atoms = Graphite("C", latticeconstant={"a": 2.46, "c": 6.7})

# Vasp calculator
calc = Vasp(
    # Command to run VASP
    command="mpirun -np 192 vasp_std",
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
    kpts=[8, 8, 8],
    gamma=True,  # Centered on Gamma point
    # Write some file or not
    lwave=False,
    lcharg=False,
    # For parallelization
    ncore=32,
)
calc.clean()
# Assign calculator tot he atoms object
atoms.calc = calc

# Phonon calculation
savedir = TARGET_DIR  # This is where we will store the cache
# Phonon calculator
ph = Phonons(atoms, calc, supercell=(NCELL, NCELL, 2), delta=0.01, name=savedir)
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

if args.correction:
    # The energy seems to off by cube-root of the volume
    energies /= atoms.get_volume() ** (1 / 3)

labels = list(bs.get_labels())
labels[2] = [r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"]

with open(TARGET_DIR / "phonon_graphite.pkl", "wb") as f:
    pickle.dump({"energies": energies, "labels": labels}, f, protocol=4)
