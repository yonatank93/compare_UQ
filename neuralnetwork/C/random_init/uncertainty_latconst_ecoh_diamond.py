#!/usr/bin/env python
# coding: utf-8

# In this notebook, I want to compute the uncertainty of the equilibrium lattice constant and the cohesive energy from the random initialization ensembles.

# In[1]:


from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
import json
import re
import argparse
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

WORK_DIR = Path().absolute()
sys.path.append(str(WORK_DIR.parent))


# In[2]:


from energyvslatconst.relaxation_latconst import equilibrate_diamond


# In[3]:


# Read settings
ROOT_DIR = WORK_DIR.parent
SETTINGS_DIR = ROOT_DIR / "settings"

# Command line argument
arg_parser = argparse.ArgumentParser("Settings file path")
arg_parser.add_argument(
    "-p", "--path", default=SETTINGS_DIR / "settings0.json", dest="settings_path"
)
args = arg_parser.parse_args()

settings_path = Path(args.settings_path)
with open(settings_path, "r") as f:
    settings = json.load(f)

RES_DIR = WORK_DIR / "results" / re.match(r"^[^_\.]+", settings_path.name).group()
if not RES_DIR.exists():
    RES_DIR.mkdir(parents=True)


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
        potential = f"DUNN_C_randinit_{set_idx:03d}"
        return equilibrate_diamond(potential, ainit)

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
