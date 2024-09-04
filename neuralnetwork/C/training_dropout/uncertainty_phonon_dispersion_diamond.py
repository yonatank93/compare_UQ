#!/usr/bin/env python
# coding: utf-8

# In this notebook, I want to compute the uncertainty of the phonon dispersion curves from the bootstrap ensembles.

# In[1]:


from pathlib import Path
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import sys
import argparse
from multiprocessing import Pool

from ase.build import bulk
from ase.calculators.kim import KIM
from ase.phonons import Phonons
from ase.visualize import view

import numpy as np
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.style.use("default")


# In[2]:


# Read settings
WORK_DIR = Path().absolute()
ROOT_DIR = WORK_DIR.parent
DATA_DIR = ROOT_DIR / "data"

# settings = {"partition": "mingjian", "Nlayers": 4, "Nnodes": [128, 128, 128]}
arg_parser = argparse.ArgumentParser("Settings of the calculations")
arg_parser.add_argument("-p", "--partition", dest="partition")
arg_parser.add_argument("-l", "--nlayers", type=int, dest="nlayers")
arg_parser.add_argument("-n", "--nnodes", nargs="+", type=int, dest="nnodes")
arg_parser.add_argument("-d", "--dropout", type=float, default=0.1, dest="dropout")
args = arg_parser.parse_args()
if len(sys.argv) > 1:
    # Command line arguments present
    settings = {
        "partition": args.partition,
        "Nlayers": args.nlayers,
        "Nnodes": args.nnodes,
        "dropout_ratio": args.dropout,
    }
else:
    # No command line arguments, read setting file
    with open(ROOT_DIR / "settings.json", "r") as f:
        settings = json.load(f)

partition = settings["partition"]
suffix = "_".join([str(n) for n in settings["Nnodes"]])
RES_DIR = (
    WORK_DIR
    / "results"
    / f"dropout_d{settings['dropout_ratio']}"
    / f"{partition}_partition_{suffix}"
)
PLOT_DIR = RES_DIR / "plots"
if not PLOT_DIR.exists():
    PLOT_DIR.mkdir(parents=True)


# In[3]:


# Diamond bulk
a0 = 3.56
atoms = bulk("C", "diamond", a0)
# view(atoms.repeat((1, 1, 1)))


# In[4]:


potential = "DUNN_best_train"
# potential = "DUNN_WenTadmor_2019v3_C__MO_714772088128_000"


def phonon_wrapper(set_idx):
    sample_dir = RES_DIR / f"{set_idx:03d}"

    # Phonon calculator
    calc = KIM(potential)
    calc.set_parameters(active_member_id=[[0], [set_idx + 1]])
    ph = Phonons(
        atoms, calc, supercell=(7, 7, 7), delta=0.1, name=sample_dir / "phonon_diamond"
    )
    ph.run()

    # Read forces and assemble the dynamical matrix
    ph.read(acoustic=True)
    # ph.clean()

    path = atoms.cell.bandpath("GXKGL", npoints=100)
    bs = ph.get_band_structure(path)
    return bs.energies


with Pool(25) as p:
    energies = np.array(list(p.map(phonon_wrapper, range(100))))

energies = energies[:, 0]


# In[5]:


# Get band structure
# Phonon calculator
calc = KIM(potential)
calc.set_parameters(active_member_id=[[0], [0]])
ph = Phonons(
    atoms,
    calc,
    supercell=(7, 7, 7),
    delta=0.1,
    name=RES_DIR / "000" / "phonon_diamond",
)
ph.run()
ph.read(acoustic=True)
path = atoms.cell.bandpath("GXKGL", npoints=100)
bs = ph.get_band_structure(path)


# In[6]:


# Convert to THz
conversion = 4.136e-3  # 1 Thz = 4.136 meV
energies /= conversion

labels = list(bs.get_labels())
labels[2] = [r"$\Gamma$", r"$X$", r"$K$", r"$\Gamma$", r"$L$"]
mean_energies = np.mean(energies, axis=0)
error_energies = np.std(energies, axis=0)


# In[7]:


# Export the data needed to plot the result
plot_data_dict = {
    "energies": {"values": energies, "metadata": "Phonon dispersion energies in THz"},
    "labels": labels,
}
with open(RES_DIR / "uncertainty_phonon_dispersion_diamond.pkl", "wb") as f:
    pickle.dump(plot_data_dict, f, protocol=4)


# In[8]:


colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
plt.figure()

for ii, eng in enumerate(mean_energies.T):
    plt.fill_between(
        labels[0],
        eng - error_energies[:, ii],
        eng + error_energies[:, ii],
        color=colors[ii],
        alpha=0.3,
    )
    plt.plot(labels[0], eng, c=colors[ii])

for xcoord, name in zip(labels[1], labels[2]):
    plt.axvline(xcoord, c="k", ls="--")
plt.xticks(labels[1], labels[2])
plt.xlim(labels[1][[0, -1]])
plt.ylim(0, 50)
plt.ylabel("Energies (THz)")
plt.savefig(PLOT_DIR / "phonon_dispersion_diamond.png", bbox_inches="tight")
# plt.show()
plt.close()


# In[ ]:
