#!/usr/bin/env python
# coding: utf-8

# In this notebook, I want to compute the uncertainty of the equilibrium lattice constant and the cohesive energy from the bootstrap ensembles.

# In[1]:


from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
import json
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.style.use("default")

WORK_DIR = Path(__file__).absolute().parent
RES_DIR = WORK_DIR / "results" / "dropout"
sys.path.append(str(WORK_DIR.parent))


# In[2]:


from energyvslatconst.relaxation_latconst import equilibrate_graphite


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
