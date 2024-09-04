#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import json
from tqdm import tqdm
import sys
from multiprocessing import Pool
import argparse

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
arg_parser.add_argument("-m", "--mode", dest="mode")
args = arg_parser.parse_args()
# Read settings from file for missing arguments
with open(ROOT_DIR / "settings.json", "r") as f:
    settings_from_file = json.load(f)
# Missing values
if args.partition is None:
    args.partition = settings_from_file["partition"]
if args.nlayers is None:
    args.nlayers = settings_from_file["Nlayers"]
if args.nnodes is None:
    args.nnodes = settings_from_file["NNodes"]
# Construct settings dictionary
settings = {"partition": args.partition, "Nlayers": args.nlayers, "Nnodes": args.nnodes}


# Directories
partition = settings["partition"]
suffix = "_".join([str(n) for n in settings["Nnodes"]])
PART_DIR = DATA_DIR / f"{partition}_partition_data"
RES_DIR = WORK_DIR / "results" / f"{partition}_partition_{suffix}" / args.mode
if not RES_DIR.exists():
    RES_DIR.mkdir()
PLOT_DIR = RES_DIR / "plots"
if not PLOT_DIR.exists():
    PLOT_DIR.mkdir()


# In[4]:


# In[5]:


##########################################################################################
# FIM
# ---
# Compute the energy ensembles
ainit = 2.466
alist = np.linspace(0.93, 1.09, 21) * ainit
preds_samples_file = RES_DIR / "uncertainty_energy_vs_latconst_graphite.npz"
if preds_samples_file.exists():
    preds_data = np.load(preds_samples_file)
    energy_ensembles = preds_data["energy_ensembles"]
    latconst_ensembles = preds_data["latconst_ensembles"]
else:
    # Use multiprocessing to speed up the calculation
    def energyvslatconst_wrapper(set_idx):
        modelname = f"DUNN_C_fimbayes_{set_idx:03d}"
        elist = energyvslatconst(modelname, alist, "graphite", 0)
        return elist

    with Pool(25) as p:
        energy_latconst_ensembles = list(
            tqdm(p.imap(energyvslatconst_wrapper, range(100)), total=100)
        )

    energy_latconst_ensembles = np.array(energy_latconst_ensembles).astype(float)
    energy_ensembles = energy_latconst_ensembles[:, 1]
    latconst_ensembles = energy_latconst_ensembles[:, 2]
    np.savez(
        preds_samples_file,
        alist=alist,
        energy_ensembles=energy_ensembles,
        latconst_ensembles=latconst_ensembles,
    )


# In[6]:


energy_mean = np.mean(energy_ensembles, axis=0)
energy_error = np.std(energy_ensembles, axis=0)

latconst_mean = np.mean(latconst_ensembles, axis=0)
latconst_error = np.std(latconst_ensembles, axis=0)


# In[7]:


# Plot the result curves
# Energy vs lattice constant
plt.figure()

# Dropout
plt.fill_between(
    alist,
    energy_mean - energy_error,
    energy_mean + energy_error,
    alpha=0.5,
    color="tab:orange",
)
plt.plot(alist, energy_mean, "-", color="tab:orange", label="DUNN dropout")
# plt.ylim(-8.12, -7.25)
plt.xlabel(r"Lattice constant $a$ $(\AA)$")
plt.ylabel("Energy (eV/atom)")
plt.savefig(PLOT_DIR / "energy_vs_latconst_graphite.png", bbox_inches="tight")
# plt.show()
plt.close()


# In[8]:


# Plot the result curves
# latconst vs lattice constant
plt.figure()

# Dropout
plt.fill_between(
    alist,
    latconst_mean - latconst_error,
    latconst_mean + latconst_error,
    alpha=0.5,
    color="tab:orange",
)
plt.plot(alist, latconst_mean, "-", color="tab:orange", label="DUNN dropout")
# plt.ylim(-8.12, -7.25)
plt.xlabel(r"Lattice constant $a$ $(\AA)$")
plt.ylabel(r"Lattice constant $c$ $(\AA)$")
plt.savefig(PLOT_DIR / "energy_vs_latconst_cva_graphite.png", bbox_inches="tight")
# plt.show()
plt.close()


# In[ ]:
