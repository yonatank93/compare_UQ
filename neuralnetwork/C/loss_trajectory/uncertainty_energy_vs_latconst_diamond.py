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

WORK_DIR = Path().absolute()
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


##########################################################################################
# Loss trajectory
# ---------------
# Compute the energy ensembles
ainit = 3.56
alist = np.linspace(0.93, 1.09, 21) * ainit
preds_samples_file = RES_DIR / "uncertainty_energy_vs_latconst_diamond.npz"
if preds_samples_file.exists():
    preds_data = np.load(preds_samples_file)
    energy_ensembles = preds_data["energy_ensembles"]
else:
    # Use multiprocessing to speed up the calculation
    def energyvslatconst_wrapper(set_idx):
        modelname = f"DUNN_C_losstraj_{set_idx:03d}"
        _, elist = energyvslatconst(modelname, alist, "diamond", 0)
        return elist

    with Pool(25) as p:
        energy_ensembles = list(p.map(energyvslatconst_wrapper, range(100)))

    energy_ensembles = energy_ensembles = np.array(energy_ensembles).astype(float)
    np.savez(preds_samples_file, alist=alist, energy_ensembles=energy_ensembles)


# In[6]:


energy_mean = np.mean(energy_ensembles, axis=0)
energy_error = np.std(energy_ensembles, axis=0)


# In[7]:


# Plot the result curves
# Energy vs lattice constant
plt.figure()

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
plt.ylim(-8.12, -7.25)
plt.xlabel(r"Lattice constant $a$ $(\AA)$")
plt.ylabel("Energy (eV/atom)")
plt.savefig(PLOT_DIR / "energy_vs_latconst_diamond.png", bbox_inches="tight")
# plt.show()
plt.close()


# In[ ]:
