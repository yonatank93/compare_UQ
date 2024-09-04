#!/usr/bin/env python
# coding: utf-8

# In this notebook, I want to compute the uncertainty of the equilibrium lattice constant and the cohesive energy from the bootstrap ensembles.

# In[1]:


from pathlib import Path
import tqdm
import sys
from multiprocessing import Pool
import argparse
import json

import numpy as np

WORK_DIR = Path().absolute()
sys.path.append(str(WORK_DIR.parent))
from energyvslatconst.relaxation_latconst import equilibrate_diamond


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


# In[4]:


a0_e0_file = RES_DIR / "uncertainty_latconst_ecoh_diamond.npz"
if a0_e0_file.exists():
    a0_e0 = np.load(a0_e0_file)
    a0_list = a0_e0["a0_list"]
    e0_list = a0_e0["e0_list"]
else:
    # Use multiprocessing to speed up the calculation
    ainit = 3.56  # From materials project for diamond structure

    def equilibrate_diamond_wrapper(set_idx):
        potential = f"DUNN_C_fimbayes_{set_idx:03d}"
        return equilibrate_diamond(potential, ainit, active_member_id=0)

    with Pool(25) as p:
        a0_e0_list = np.array(list(p.map(equilibrate_diamond_wrapper, range(100))))

    a0_list, e0_list = a0_e0_list.T
    np.savez(a0_e0_file, a0_list=a0_list, e0_list=e0_list)


# In[5]:


# Lattice constant
print("Lattice constant")
print("Mean:", np.mean(a0_list))
print("Stdev:", np.std(a0_list))


# In[6]:


# Cobesive energy
print("Cohesive energy")
print("Mean:", np.mean(e0_list))
print("Stdev:", np.std(e0_list))


# In[ ]:
