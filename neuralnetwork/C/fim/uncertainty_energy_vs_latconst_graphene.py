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


# DFT data
dft_data = np.loadtxt("../energyvslatconst/dft_data.txt", delimiter=",")


##########################################################################################
# FIM
# ---
ainit = 2.466
alist = np.linspace(0.93, 1.09, 21) * ainit
preds_samples_file = RES_DIR / "uncertainty_energy_vs_latconst_graphene.npz"
if preds_samples_file.exists():
    preds_data = np.load(preds_samples_file)
    preds_samples = preds_data["energy_ensembles"]
else:
    preds_samples = np.empty((0, len(alist)))
    for ii in tqdm(range(100)):
        # Predictions
        modelname = f"DUNN_C_fimbayes_{ii:03d}"
        _, eng = energyvslatconst(modelname, alist, "graphene", 0)
        preds_samples = np.row_stack((preds_samples, eng))
    preds_samples = preds_samples.astype(float)
    np.savez(preds_samples_file, alist=alist, energy_ensembles=preds_samples)


# In[7]:


energy_mean = np.mean(preds_samples, axis=0)
energy_error = np.std(preds_samples, axis=0)


# In[8]:


# Plot
plt.figure()
plt.plot(*(dft_data.T), "r.", zorder=10, label="DFT")
# FIM
plt.fill_between(
    alist,
    energy_mean + energy_error,
    energy_mean - energy_error,
    alpha=0.5,
    color="tab:purple",
)
plt.plot(alist, energy_mean, color="tab:purple", label="DUNN FIM")

plt.ylim(-8.1, -7.5)
plt.xlabel(r"Lattice constant $a$ $(\AA)$")
plt.ylabel("Energy (eV/atom)")
# plt.legend()
plt.savefig(PLOT_DIR / "energy_vs_latconst_graphene.png", bbox_inches="tight")
# plt.show()
plt.close()


# In[9]:


plt.figure()
plt.plot(alist, energy_error)
plt.xlabel(r"Lattice constant $a$ $(\AA)$")
plt.ylabel("Energy uncertainty")
plt.savefig(
    PLOT_DIR / "energy_vs_latconst_energy_uncertainty_graphene.png", bbox_inches="tight"
)
# plt.show()
plt.close()


# In[ ]:
