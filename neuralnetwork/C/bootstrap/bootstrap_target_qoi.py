from pathlib import Path
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from energyvslatconst import energyvslatconst, equilibrate_graphene

WORK_DIR = Path(__file__).absolute().parent


# Initial setup
# Directories
WORK_DIR = Path(__file__).absolute().parent

# DFT data
dft_data = np.loadtxt(WORK_DIR / "energyvslatconst_dft_data.txt", delimiter=",")


##########################################################################################
# Dropout
# -------

potential = "DUNN_best_train"
ainit = 2.466
a0, e0 = equilibrate_graphene(potential, ainit)
alist = np.linspace(0.93, 1.08, 29) * a0  # np.linspace(0.93, 1.07, 16) * a0
_, energy_mean_do, energy_error_do = energyvslatconst(potential, alist)


##########################################################################################
# Bootstrap
# ---------
# Compute the energy ensembles
energy_ensembles = np.empty((0, len(alist)))
for set_idx in tqdm(range(100)):
    RES_DIR = WORK_DIR / "results" / "bootstrap" / f"{set_idx:03d}"
    modelname = f"DUNN_C_bootstrap_{set_idx:03d}"

    if (RES_DIR / f"DUNN_C_bootstrap_{set_idx:03d}").exists():
        _, elist, _ = energyvslatconst(modelname, alist, apply_dropout=False)
        energy_ensembles = np.row_stack((energy_ensembles, elist))
    else:
        continue
energy_ensembles = energy_ensembles.astype(float)


energy_mean_bs = np.mean(energy_ensembles, axis=0)
energy_error_bs = np.std(energy_ensembles, axis=0)


# Plot the result curves
# Energy vs lattice constant
plt.figure()
plt.plot(*(dft_data.T), "r.", zorder=10, label="DFT")
plt.fill_between(
    alist,
    energy_mean_bs - energy_error_bs,
    energy_mean_bs + energy_error_bs,
    alpha=0.5,
    color="tab:blue",
)
plt.plot(alist, energy_mean_bs, "-", color="tab:blue", label="DUNN bootstrap")
plt.fill_between(
    alist,
    energy_mean_do - energy_error_do,
    energy_mean_do + energy_error_do,
    alpha=0.5,
    color="tab:orange",
)
plt.plot(alist, energy_mean_do, "-", color="tab:orange", label="DUNN dropout")
plt.ylim(-8.1, -7.5)
plt.xlabel(r"Lattice constant $a$ $(\AA)$")
plt.ylabel("Energy (eV/atom)")
plt.legend()

plt.show()
