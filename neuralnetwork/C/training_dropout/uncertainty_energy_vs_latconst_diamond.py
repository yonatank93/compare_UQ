#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import json
from tqdm import tqdm
import sys
import argparse
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.style.use("default")

WORK_DIR = Path(__file__).absolute().parent
sys.path.append(str(WORK_DIR.parent))
from energyvslatconst import energyvslatconst


# In[2]:


# In[3]:


# Read settings
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


# In[4]:


##########################################################################################
# Dropout
# -------
# Compute the energy ensembles
potential = "DUNN_best_train"
ainit = 3.56
alist = np.linspace(0.93, 1.09, 21) * ainit
preds_samples_file = RES_DIR / "uncertainty_energy_vs_latconst_diamond.npz"
if preds_samples_file.exists():
    preds_data = np.load(preds_samples_file)
    energy_ensembles = preds_data["energy_ensembles"]
else:
    # Use multiprocessing to speed up the calculation
    def energyvslatconst_wrapper(set_idx):
        _, elist = energyvslatconst(potential, alist, "diamond", set_idx + 1)
        return elist

    with Pool(25) as p:
        energy_ensembles = list(p.map(energyvslatconst_wrapper, range(100)))

    energy_ensembles = energy_ensembles = np.array(energy_ensembles).astype(float)
    energy_mean_do = np.mean(energy_ensembles, axis=0)
    energy_error_do = np.std(energy_ensembles, axis=0)
    np.savez(preds_samples_file, alist=alist, energy_ensembles=energy_ensembles)


# In[5]:


energy_mean_do = np.mean(energy_ensembles, axis=0)
energy_error_do = np.std(energy_ensembles, axis=0)


# In[6]:


# Plot the result curves
# Energy vs lattice constant
plt.figure()

# Dropout
plt.fill_between(
    alist,
    energy_mean_do - energy_error_do,
    energy_mean_do + energy_error_do,
    alpha=0.5,
    color="tab:orange",
)
plt.plot(alist, energy_mean_do, "-", color="tab:orange", label="DUNN dropout")
plt.ylim(-8.12, -7.25)
plt.xlabel(r"Lattice constant $a$ $(\AA)$")
plt.ylabel("Energy (eV/atom)")
plt.savefig(PLOT_DIR / "energy_vs_latconst_diamond.png", bbox_inches="tight")
# plt.show()
plt.close()


# In[ ]:
