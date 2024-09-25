"""I want to use this script to evaluate the loss in each saved epoch against the training
or test data (specified through the CLI argument).
"""

##########################################################################################
from pathlib import Path
import re
import json
import sys
import shutil
import subprocess
import argparse

import numpy as np
import torch

from kliff import nn
from kliff.calculators import CalculatorTorch
from kliff.dataset import Dataset
from kliff.dataset.weight import Weight
from kliff.descriptors import SymmetryFunction
from kliff.loss import Loss
from kliff.models import NeuralNetwork

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


# Read settings
# Dataset
dataset_path = Path(settings["dataset"]["dataset_path"])  # Path to the dataset
FP_DIR = Path(settings["dataset"]["fingerprints_path"])  # Path to the fingerprint files

# Architecture
Nlayers = settings["architecture"]["Nlayers"]  # Number of layers, including output
Nnodes = settings["architecture"]["Nnodes"]  # Number of nodes for each hidden layer
dropout_ratio = settings["architecture"]["dropout_ratio"]  # Dropout ratio

# Optimizer settings
batch_size = 1  # Set batch size 1 so that we won't divide by batch size
# We use the first learning rate up to epoch number listed as the first element of
# nepochs_list. Then, we use the second learning rate up to the second nepochs.
lr_list = settings["optimizer"]["learning_rate"]
nepochs_list = settings["optimizer"]["nepochs"]

nepochs_initial = 2000  # Run this many epochs first before start saving the model
nepochs_save_period = 10  # Then run and save every this many epochs
nepochs_total = nepochs_list[-1]  # How many epochs to run in total

# Directories to store results
RES_DIR = WORK_DIR / "results" / re.match(r"^[^_\.]+", settings_path.name).group()
if not RES_DIR.exists():
    RES_DIR.mkdir(parents=True)
MODEL_DIR = RES_DIR / "models"


##########################################################################################
# Model
# -----

# Descriptor
descriptor = SymmetryFunction(
    cut_name="cos", cut_dists={"C-C": 5.0}, hyperparams="set51", normalize=True
)
model = NeuralNetwork(descriptor)

# Layers
hidden_layer_mappings = []
for ii in range(Nlayers - 2):
    hidden_layer_mappings.append(nn.Dropout(dropout_ratio))
    hidden_layer_mappings.append(nn.Linear(Nnodes[ii], Nnodes[ii + 1]))
    hidden_layer_mappings.append(nn.Tanh())

model.add_layers(
    # input layer
    nn.Linear(descriptor.get_size(), Nnodes[0]),  # Mapping from input layer to the first
    nn.Tanh(),  # hidden layer
    # hidden layer(s)
    *hidden_layer_mappings,  # Mappings between hidden layers in the middle
    # hidden layer(s)
    nn.Dropout(dropout_ratio),  # Mapping from the last hidden layer to the output layer
    nn.Linear(Nnodes[-1], 1),
    # output layer
)


##########################################################################################
# Training set and calculator
# ---------------------------

# training set
if "train" in dataset_path.name:
    config_id = "train"
elif "test" in dataset_path.name:
    config_id = "test"
weight = Weight(energy_weight=1.0, forces_weight=np.sqrt(0.1))
tset = Dataset(dataset_path, weight)
configs = tset.get_configs()
nconfigs = len(configs)

# calculator
gpu = False
calc = CalculatorTorch(model, gpu=gpu)
_ = calc.create(
    configs,
    nprocs=20,
    reuse=True,
    fingerprints_filename=FP_DIR / f"fingerprints_{config_id}.pkl",
    fingerprints_mean_stdev_filename=FP_DIR
    / f"fingerprints_{config_id}_mean_and_stdev.pkl",
)
loader = calc.get_compute_arguments(batch_size)


##########################################################################################
# Loss function
# -------------

residual_data = {"normalize_by_natoms": True}
loss = Loss(calc, residual_data=residual_data)

##########################################################################################
# Training
# --------

loss_values_file = RES_DIR / f"loss_values_{config_id}.txt"
if loss_values_file.exists():
    loss_values = np.loadtxt(loss_values_file)
else:
    loss_values = np.empty((0, 2))

if nepochs_initial not in loss_values[:, 0]:
    # Evaluate the model at the end of the bur-in period.
    trained_model_file = MODEL_DIR / f"model_epoch{nepochs_initial}.pkl"
    model.load(trained_model_file)
    loss_values = np.vstack(
        (loss_values, [nepochs_initial, loss._get_loss_epoch(loader)])
    )

# Continue evaluating the loss for the rest of the training trajectory.
ii = 0
nepochs_done = nepochs_initial
while nepochs_done < nepochs_total:
    try:
        start_epoch = nepochs_initial + ii * nepochs_save_period + 1
        num_epochs = nepochs_save_period - 1
        nepochs_done = start_epoch + num_epochs

        if nepochs_done not in loss_values[:, 0]:
            trained_model_file = MODEL_DIR / f"model_epoch{nepochs_done}.pkl"
            model.load(trained_model_file)
            loss_values = np.vstack(
                (loss_values, [nepochs_done, loss._get_loss_epoch(loader)])
            )
        ii += 1
        print(start_epoch, nepochs_done)
    except Exception as e:
        print(e)
        break

np.savetxt(loss_values_file, loss_values)


# Best model
idx = np.argmin(loss_values[:, 1])
best_epoch = int(loss_values[idx, 0])
best_model_file = MODEL_DIR / f"model_epoch{best_epoch}.pkl"
model.load(best_model_file)
# Move the best model pickle file
shutil.copy(best_model_file, RES_DIR / f"model_best_{config_id}.pkl")

# Write KIM model
kim_model_file = RES_DIR / f"DUNN_best_{config_id}"
model.write_kim_model(kim_model_file)
# # (Re)Install best model
# subprocess.run(
#     ["kim-api-collections-management", "remove", "--force", kim_model_file.name]
# )
# subprocess.run(["kim-api-collections-management", "install", "user", str(kim_model_file)])
