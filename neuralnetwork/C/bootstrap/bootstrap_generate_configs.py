from pathlib import Path
import json
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
from kliff.uq.bootstrap import BootstrapNeuralNetworkModel

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
batch_size = settings["optimizer"]["batch_size"]
# We use the first learning rate up to epoch number listed as the first element of
# nepochs_list. Then, we use the second learning rate up to the second nepochs.
lr_list = settings["optimizer"]["learning_rate"]
nepochs_list = settings["optimizer"]["nepochs"]

nepochs_initial = 2000  # Run this many epochs first before start saving the model
nepochs_save_period = 10  # Then run and save every this many epochs
nepochs_total = nepochs_list[-1]  # How many epochs to run in total

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


##########################################################################################
# Loss function
# -------------

residual_data = {"normalize_by_natoms": True}
loss = Loss(calc, residual_data=residual_data)

##########################################################################################
# Bootstrap

# Instantiate
BS = BootstrapNeuralNetworkModel(loss, seed=seed)

# Try to load the data from previous calculations
# Bootstrap compute arguments samples
nsamples_target = 100
bootstrap_fingerprints_file = Path(RES_DIR / f"bootstrap_fingerprints.json")
if bootstrap_fingerprints_file.exists():
    BS.load_bootstrap_compute_arguments(bootstrap_fingerprints_file)
else:
    BS.generate_bootstrap_compute_arguments(nsamples_target)
    BS.save_bootstrap_compute_arguments(bootstrap_fingerprints_file)
