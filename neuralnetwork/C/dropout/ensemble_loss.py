"""Compute the ensemble training loss, i.e., the training loss for each ensemble member.
"""

##########################################################################################
from pathlib import Path
from datetime import datetime
import json
import argparse
import subprocess
from pprint import pprint
from tqdm import tqdm

import numpy as np
import torch

from kliff import nn
from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.dataset.weight import Weight
from kliff.models import KIMModel
from kliff.loss import Loss, energy_forces_residual
from kliff import parallel

# Random seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_tensor_type(torch.DoubleTensor)


##########################################################################################
# Initial Setup
# -------------

WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
SETTINGS_DIR = ROOT_DIR / "settings"

# Read command line argument
arg_parser = argparse.ArgumentParser("Settings of the calculations")
arg_parser.add_argument(
    "-s", "--settings-path", default=SETTINGS_DIR / "settings0.json", dest="settings_path"
)
args = arg_parser.parse_args()
settings_path = Path(args.settings_path)
# Load settings from json file
with open(settings_path, "r") as f:
    settings = json.load(f)
print("Train a NN model with the following settings:")
pprint(settings)


# Read settings
# Dataset
dataset_path = Path(settings["dataset"]["dataset_path"])  # Path to the dataset
FP_DIR = Path(settings["dataset"]["fingerprints_path"])  # Path to the fingerprint files

# Directories to store results
RES_DIR = WORK_DIR / "results" / settings_path.with_suffix("").name
if not RES_DIR.exists():
    RES_DIR.mkdir(parents=True)
MODEL_DIR = (
    ROOT_DIR
    / "training"
    / "results"
    / settings_path.with_suffix("").name
    / "DUNN_best_train"
)


##########################################################################################
# Model and Loss function
# -----------------------

# Install the model
subprocess.run(
    "kim-api-collections-management remove --force DUNN_best_train", shell=True
)
subprocess.run(f"kim-api-collections-management install user {MODEL_DIR}", shell=True)

# Model
model = KIMModel("DUNN_best_train")

# training set
weight = Weight(energy_weight=1.0, forces_weight=np.sqrt(0.1))
tset = Dataset(dataset_path, weight)
configs = tset.get_configs()
nconfigs = len(configs)

# calculator
calc = Calculator(model)
_ = calc.create(configs, use_energy=True, use_forces=True)
cas = calc.get_compute_arguments()

# Loss function
residual_data = {"normalize_by_natoms": True}
loss = Loss(calc, residual_data=residual_data)


##########################################################################################
# Compute ensemble loss
# ---------------------

if "train" in dataset_path.name:
    ensemble_loss_file = RES_DIR / "ensemble_loss_train.txt"
elif "test" in dataset_path.name:
    ensemble_loss_file = RES_DIR / "ensemble_loss_test.txt"

if not ensemble_loss_file.exists():
    ensemble_loss = []

    for ii in tqdm(range(100)):
        # Update model parameter - set which dropout matrix to use
        calc.model.kim_model.set_parameter(1, 0, ii)  # pidx, array_idx, update_value
        # Compute residuals
        nprocs = 50
        residuals = parallel.parmap2(
            loss._get_residual_single_config,
            cas,
            calc,
            energy_forces_residual,
            residual_data,
            nprocs=nprocs,
            tuple_X=False,
        )
        residual = np.concatenate(residuals)
        ensemble_loss = np.append(ensemble_loss, 0.5 * np.linalg.norm(residual) ** 2)
    np.savetxt(ensemble_loss_file, ensemble_loss)
