#!/usr/bin/env python
# coding: utf-8

# In this notebook, I want to compute the uncertainty of the equilibrium lattice constant and the cohesive energy from the bootstrap ensembles.

# In[1]:


from pathlib import Path
from datetime import datetime
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
from energyvslatconst.relaxation_latconst import equilibrate_graphite


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


a0_c0_e0_file = RES_DIR / "uncertainty_latconst_ecoh_graphite.npz"
if a0_c0_e0_file.exists():
    a0_c0_e0 = np.load(a0_c0_e0_file)
    a0_list = a0_c0_e0["a0_list"]
    c0_list = a0_c0_e0["c0_list"]
    e0_list = a0_c0_e0["e0_list"]
else:
    a0_list = []
    c0_list = []
    e0_list = []

    ainit = 2.466  # From materials project for graphite-like structure
    cinit = 3.348  # From https://doi.org/10.1073/pnas.2134173100
    for ii in tqdm(range(100)):
        # Equilibration
        a0, c0, e0 = equilibrate_graphite(
            "DUNN_best_train", ainit, cinit, active_member_id=ii + 1
        )
        a0_list = np.append(a0_list, a0)
        c0_list = np.append(c0_list, c0)
        e0_list = np.append(e0_list, e0)

    np.savez(a0_c0_e0_file, a0_list=a0_list, c0_list=c0_list, e0_list=e0_list)


# In[5]:


# Lattice constant a
print("In-plane lattice constant a")
print("Mean:", np.mean(a0_list))
print("Stdev:", np.std(a0_list))


# In[6]:


# Lattice constant c
print("Out-of-plane lattice constant c")
print("Mean:", np.mean(c0_list))
print("Stdev:", np.std(c0_list))


# In[7]:


# Cobesive energy
print("Cohesive energy")
print("Mean:", np.mean(e0_list))
print("Stdev:", np.std(e0_list))


# In[ ]:
