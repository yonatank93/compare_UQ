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
import json
import argparse
from tqdm import tqdm
from pprint import pprint

import numpy as np
import torch

from kliff import nn
from kliff.calculators import CalculatorTorch
from kliff.dataset import Dataset
from kliff.dataset.weight import Weight
from kliff.descriptors import SymmetryFunction
from kliff.loss import Loss
from kliff.models import NeuralNetwork

np.random.seed(1)

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

# Architecture
Nlayers = settings["architecture"]["Nlayers"]  # Number of layers, including output
Nnodes = settings["architecture"]["Nnodes"]  # Number of nodes for each hidden layer
dropout_ratio = settings["architecture"]["dropout_ratio"]  # Dropout ratio

# Optimizer settings
nepochs_list = settings["optimizer"]["nepochs"]
nepochs_total = nepochs_list[-1]

# Directories to store results
RES_DIR = WORK_DIR / "results" / settings_path.with_suffix("").name
if not RES_DIR.exists():
    RES_DIR.mkdir(parents=True)


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
    fingerprints_filename=FP_DIR / f"fingerprints_train.pkl",
    fingerprints_mean_stdev_filename=FP_DIR / f"fingerprints_train_mean_and_stdev.pkl",
)
loader = calc.get_compute_arguments(1)


##########################################################################################
# Compute Loss for Last Epoch
# ---------------------------

print("Compute the loss of the last epoch")
residual_data = {"normalize_by_natoms": True}

# Compute loss value of the last epoch for each ensemble model
loss_file = RES_DIR / "loss_last_epoch.txt"
nensemble = 100
if not loss_file.exists():
    # First column will be used to store the ensemble id, basically just number from 0 to
    # 99. Second column will be used to store the loss value.
    loss_last_epoch = np.zeros((nensemble, 2))
    for ii in tqdm(range(nensemble)):
        # ii = 0
        sample_dir = RES_DIR / f"{ii:03d}"
        # Load and update the weights and biases of the last epoch
        last_params = np.load(sample_dir / "last_params.npy")
        calc.update_model_params(last_params)
        # Loss evaluation
        loss = Loss(calc, residual_data=residual_data)
        loss_value = loss._get_loss_epoch(loader)
        loss_last_epoch[ii] = [ii, loss_value]
    np.savetxt(loss_file, loss_last_epoch, fmt="%d %.16e")
else:
    loss_last_epoch = np.loadtxt(loss_file)
