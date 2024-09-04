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
args = arg_parser.parse_args()
if len(sys.argv) > 1:
    # Command line arguments present
    settings = {
        "partition": args.partition,
        "Nlayers": args.nlayers,
        "Nnodes": args.nnodes,
    }
else:
    # No command line arguments, read setting file
    with open(ROOT_DIR / "settings.json", "r") as f:
        settings = json.load(f)

partition = settings["partition"]
suffix = "_".join([str(n) for n in settings["Nnodes"]])
RES_DIR = WORK_DIR / "results" / f"{partition}_partition_{suffix}"
PLOT_DIR = RES_DIR / "plots"
if not PLOT_DIR.exists():
    PLOT_DIR.mkdir(parents=True)


# In[4]:


# DFT data
dft_data = np.loadtxt("../energyvslatconst/dft_data.txt", delimiter=",")


# In[5]:


##########################################################################################
# Loss trajectory
# ---------------
# Compute the energy ensembles
ainit = 2.466
alist = np.linspace(0.93, 1.09, 21) * ainit
preds_samples_file = RES_DIR / "uncertainty_energy_vs_latconst_graphene.npz"
if preds_samples_file.exists():
    preds_data = np.load(preds_samples_file)
    energy_ensembles = preds_data["energy_ensembles"]
else:
    energy_ensembles = np.empty((0, len(alist)))
    for set_idx in tqdm(range(100)):
        # Predictions
        modelname = f"DUNN_C_losstraj_{set_idx:03d}"
        _, elist = energyvslatconst(modelname, alist, "graphene", 0)
        energy_ensembles = np.row_stack((energy_ensembles, elist))
    energy_ensembles = energy_ensembles.astype(float)
    np.savez(preds_samples_file, alist=alist, energy_ensembles=energy_ensembles)


# In[7]:


energy_mean = np.mean(energy_ensembles, axis=0)
energy_error = np.std(energy_ensembles, axis=0)


# In[8]:


# Plot the result curves
# Energy vs lattice constant
plt.figure()
plt.plot(*(dft_data.T), "r.", zorder=10, label="DFT")

# Loss trajectory
plt.fill_between(
    alist,
    energy_mean - energy_error,
    energy_mean + energy_error,
    alpha=0.5,
    color="g",
    zorder=10,
)
plt.plot(alist, energy_mean, "-", color="g", label="DUNN random init")
plt.ylim(-8.1, -7.5)
plt.xlabel(r"Lattice constant $a$ $(\AA)$")
plt.ylabel("Energy (eV/atom)")
# plt.legend()
plt.savefig(PLOT_DIR / "energy_vs_latconst_graphene.png", bbox_inches="tight")
# plt.show()
plt.close()


# In[ ]:
