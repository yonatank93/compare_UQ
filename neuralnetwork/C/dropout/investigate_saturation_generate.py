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
from collections import defaultdict
from copy import deepcopy

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

# Load the parameters
TRAIN_RES_DIR = ROOT_DIR / "training" / Path(*RES_DIR.parts[-3:-1])
model_params_file = TRAIN_RES_DIR / "model_best_train.pkl"
model.load(model_params_file)


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

# Retrieve the parameters of the last layer
last_params = deepcopy(calc.get_opt_params()[-Nnodes[-1] - 1 :])

# Retrieve the dropout matrices
MODEL_DIR = TRAIN_RES_DIR / "DUNN_best_train"
dropout_binary_file = MODEL_DIR / "dropout_binary.params"
with open(dropout_binary_file, "r") as f:
    lines = f.readlines()
dropout_data = defaultdict(lambda: defaultdict(list))
instance = None
layer = None

for ii, line in enumerate(lines):
    line = line.strip()
    # Find the number of ensemble members
    if match := re.match(r"(\d+)\s+#\s*ensemble size", line):
        ensemble_size = int(match.group(1))  # Extract the number of ensemble members
        continue  # Skip to the next line

    if match := re.match(r"# instance (\d+)", line):
        instance = int(match.group(1))

    elif match := re.match(r"# layer (\d+)", line):
        layer = int(match.group(1))

    elif instance is not None and layer is not None:
        # Parse binary dropout parameters
        do_diag = map(float, line.split())
        dropout_data[instance][layer].extend(do_diag)
        layer = None

# Compute the output of the last activation function (i.e., the input to the output layer)
model.layers.pop()  # Remove the last linear layer

# Replace Dropout layers with Identity layers - Otherwise, dropout will be applied twice
# and incorrectly
for ii, layer in enumerate(model.layers):
    if isinstance(layer, nn.Dropout):
        model.layers[ii] = nn.Identity()  # Dropout is now manually handled


def model_forward(x, set_idx):
    """Custom model.forward method that applies the retrieved dropout matrices and ignore
    the last linear layer
    """
    # No dropout applied to input layer
    ido = 1  # Index of dropout layer
    for layer in model.layers:
        x = layer.forward(x)
        if isinstance(layer, nn.Identity):
            dropout_mask = torch.tensor(
                dropout_data[set_idx][ido], dtype=torch.float32, device=x.device
            )
            x = x * dropout_mask
            ido += 1
    return x


output = []
for fp in tqdm(fingerprints):
    zeta = torch.tensor(fp["zeta"], dtype=torch.float32)
    output.append(model_forward(zeta, args.set_idx).detach().numpy())
# Export
with open(RES_DIR / f"input_last_layer_{structure}.pkl", "wb") as f:
    pickle.dump(output, f)

# Output the parameters of the last layer after applying dropout
last_params_with_do = last_params * np.append(dropout_data[args.set_idx][Nlayers - 1], 1)
np.save(RES_DIR / f"last_layer_params.npy", last_params_with_do)
