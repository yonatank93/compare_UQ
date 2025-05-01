"""I want to use this script to focus on training NN potential with dropout. I want to
make this script to be able to take different configurations, e.g., number of hidden
layers and nodes per hidden layer. Also, I want to train the model with different numbers
of epochs.
Mingjian mentioned that when he trained his model, he didn't necessarily take the last
epochs as the final model. Instead, he search through the epochs that gave him the lowest
loss/rmse.
"""

##########################################################################################
from pathlib import Path
from datetime import datetime
import sys
import json
import os
import re
import argparse
import pickle
from tqdm import tqdm

import numpy as np
import torch

from kliff import nn
from kliff.calculators import CalculatorTorch
from kliff.dataset import Dataset
from kliff.dataset.weight import Weight
from kliff.descriptors import SymmetryFunction
from kliff.loss import Loss
from kliff.models import NeuralNetwork
from kliff.uq import BootstrapNeuralNetworkModel

# Random seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
# torch.set_default_tensor_type(torch.DoubleTensor)


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
arg_parser.add_argument("-c", "--crystal", dest="crystal")
arg_parser.add_argument("-i", "--set-idx", type=int, dest="set_idx")
args = arg_parser.parse_args()
settings_path = Path(args.settings_path)
set_idx = args.set_idx
print("Ensemble member id:", set_idx)
# Load settings from json file
with open(settings_path, "r") as f:
    settings = json.load(f)
print("Train a NN model with the following settings:")
print(json.dumps(settings, indent=4))


# Read settings
# Architecture
Nlayers = settings["architecture"]["Nlayers"]  # Number of layers, including output
Nnodes = settings["architecture"]["Nnodes"]  # Number of nodes for each hidden layer
dropout_ratio = settings["architecture"]["dropout_ratio"]  # Dropout ratio

# Directories to store results
RES_DIR = WORK_DIR / "results" / settings_path.with_suffix("").name / f"{set_idx:03d}"


##########################################################################################
# Model
# -----

# Descriptor - We skip normalization, so that we can later normalize it ourselves with
# the same normalization constants as used in the training set
descriptor = SymmetryFunction(
    cut_name="cos", cut_dists={"C-C": 5.0}, hyperparams="set51", normalize=False
)
model = NeuralNetwork(descriptor)

# Layers
hidden_layer_mappings = []
for ii in range(Nlayers - 2):
    # hidden_layer_mappings.append(nn.Dropout(dropout_ratio))
    hidden_layer_mappings.append(nn.Linear(Nnodes[ii], Nnodes[ii + 1]))
    hidden_layer_mappings.append(nn.Tanh())

model.add_layers(
    # input layer
    nn.Linear(descriptor.get_size(), Nnodes[0]),  # Mapping from input layer to the first
    nn.Tanh(),  # hidden layer
    # hidden layer(s)
    *hidden_layer_mappings,  # Mappings between hidden layers in the middle
    # hidden layer(s)
    # nn.Dropout(dropout_ratio),  # Mapping from the last hidden layer to the output layer
    nn.Linear(Nnodes[-1], 1),
    # output layer
)


##########################################################################################
# Calculator
# ----------

# Configurations
structure = args.crystal
CONFIG_DIR = ROOT_DIR / f"energyvslatconst/dft_data/xyz_files/{structure}"
tset = Dataset(CONFIG_DIR)
configs = tset.get_configs()
nconfigs = len(configs)

# calculator
gpu = False
calc = CalculatorTorch(model, gpu=gpu)
_ = calc.create(
    configs,
    use_forces=False,
    nprocs=20,
    reuse=False,
    fingerprints_filename=CONFIG_DIR / f"fingerprints.pkl",
)
# Load normalization constant
FP_DIR = Path(settings["dataset"]["fingerprints_path"])  # Path to the fingerprint files
with open(FP_DIR / "fingerprints_train_mean_and_stdev.pkl", "rb") as f:
    mean_std = pickle.load(f)
mean = mean_std["mean"]
std = mean_std["stdev"]
# Normalize the fingerprints
fingerprints = calc.get_fingerprints()
for fp in fingerprints:
    zeta = fp["zeta"].copy()
    fp["zeta"] = (zeta - mean) / std


##########################################################################################
# Saturation analysis
# -------------------

# Load the parameters
last_params = np.load(RES_DIR / "last_params.npy")
calc.update_model_params(last_params)

# Remove the last 2 layers of the model, so that the output of the forward function is
# the output of the last activation function
# model.layers.pop()
model.layers.pop()

# Compute the output of the last activation function (i.e., the input to the output layer)
output = []
for fp in tqdm(fingerprints):
    zeta = torch.tensor(fp["zeta"], dtype=torch.float32)
    output.append(model.forward(zeta).detach().numpy())
# Export
with open(RES_DIR / f"input_last_layer_{structure}.pkl", "wb") as f:
    pickle.dump(output, f)
