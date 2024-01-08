#!/usr/bin/env python
# coding: utf-8

# In this notebook, I want to assess the accuracy of the trained model.
# This is done by evaluating the energy of configurations in the test set.
# The predicted energies are compared to the DFT calculated energies in a parity plot.
#
# Additionally, I also want to look at the uncertainty of the energy predictions, evaluated using different member in the ensemble.

# In[1]:


from pathlib import Path
import json
import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read
from ase.calculators.kim import KIM

# get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("default")

# Read setting file
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
with open(ROOT_DIR / "settings.json", "r") as f:
    settings = json.load(f)
partition = settings["partition"]
PART_DIR = ROOT_DIR / f"{partition}_partition_data"
RES_DIR = WORK_DIR / "results" / f"{partition}_partition"


# # Get the reference energy data

# In[2]:


# Get all test configurations
configs_dict = {}

dataset_dir = PART_DIR / "carbon_test_set"
structures = os.listdir(dataset_dir)
for struct in structures:
    configs_dict.update({struct: {"identifier": []}})
    subdir = dataset_dir / struct
    if os.path.isdir(subdir / os.listdir(subdir)[0]):
        # The configurations are stored inside the subsubdirectory
        substructures = os.listdir(subdir)
        for substruct in substructures:
            configs = os.listdir(subdir / substruct)
            configs_paths = [str(subdir / substruct / cc) for cc in configs]
            configs_dict[struct]["identifier"].extend(configs_paths)
    else:
        # The configurations are stored inside the subdirectory
        configs = os.listdir(subdir)
        configs_paths = [str(subdir / cc) for cc in configs]
        configs_dict[struct]["identifier"].extend(configs_paths)


# In[3]:


# Get the reference data
for struct in structures:
    identifiers = configs_dict[struct]["identifier"]
    energies = []
    natoms = []
    for path in identifiers:
        atoms = read(path)
        energies = np.append(energies, atoms.info["Energy"])
        natoms = np.append(natoms, atoms.get_global_number_of_atoms())
    configs_dict[struct].update({"energy": energies, "natoms": natoms})


# # Compute energy

# In[4]:


uncertainty_energy_file = RES_DIR / "uncertainty_energy.pkl"
if uncertainty_energy_file.exists():
    with open(uncertainty_energy_file, "rb") as f:
        configs_dict = pickle.load(f)
else:
    for struct in structures:
        identifiers = configs_dict[struct]["identifier"]
        nid = len(identifiers)
        preds = np.zeros((100, nid))
        for ii, path in tqdm(enumerate(identifiers), desc=struct, total=nid):
            atoms = read(path)
            for jj in tqdm(range(100)):
                atoms.calc = KIM(f"DUNN_C_fimbayes_{jj:03d}")
                preds[jj, ii] = atoms.get_potential_energy()
        configs_dict[struct].update({"prediction": preds})

        with open(uncertainty_energy_file, "wb") as f:
            pickle.dump(configs_dict, f, protocol=4)


# In[ ]:
