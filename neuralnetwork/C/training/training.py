"""This script focuses on training a NN potential with dropout with the entire dataset.
The settings of the calculation, which include the dataset, model architecture, and
optimizer settings, can be specified by creating a JSON file and specify the path to the
JSON file as a command line argument. 

As a default, if no user-defined settings are given, we train a model with 3 hidden layers
and 128 nodes in each layer. We also apply dropout on each layer, except the input. The
model is trained against the dataset that Mingjian provided. We use batch size of 100 in
the training. The optimization is done using Adam optimizer with learning rate 1e-3 for
the first 5,000 epochs and 1e-4 for the rest of the training upto 40,000 epochs.

Mingjian also mentioned that when he trained his model, he didn't necessarily take the
last epochs as the final model. Instead, he search through the epochs that gave him the
lowest loss/rmse. So, we will also do the same, but it will be done using different
script.
"""

##########################################################################################
from pathlib import Path
from datetime import datetime
import json
import argparse
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

# Architecture
Nlayers = settings["architecture"]["Nlayers"]  # Number of layers, including output
Nnodes = settings["architecture"]["Nnodes"]  # Number of nodes for each hidden layer
dropout_ratio = settings["architecture"]["dropout_ratio"]  # Dropout ratio

# Optimizer settings
# We use the first learning rate up to epoch number listed as the first element of
# nepochs_list. Then, we use the second learning rate up to the second nepochs.
lr_list = settings["optimizer"]["learning_rate"]
bsize_list = settings["optimizer"]["batch_size"]
nepochs_list = settings["optimizer"]["nepochs"]
assert len(lr_list) == len(bsize_list) == len(nepochs_list)

# nepochs_initial = 2000  # Run this many epochs first before start saving the model
nepochs_initial = 0  # Run this many epochs first before start saving the model
nepochs_save_period = 10  # Then run and save every this many epochs
nepochs_total = nepochs_list[-1]  # How many epochs to run in total

# Directories to store results
RES_DIR = WORK_DIR / "results" / settings_path.with_suffix("").name
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
model.set_save_metadata(
    prefix=MODEL_DIR, start=nepochs_initial, frequency=nepochs_save_period
)

# In Mingjian's original script, the initial values of the weights were generated using
# xavier normal function, while the bias were set to zero initially. However, pytorch
# default initialization uses different functions, unless if we specify them after we
# initialize the layers. That is exactly what I want to try here.
layers = model.layers
for layer in layers:
    if isinstance(layer, nn.Linear):
        # Initialize the weights using xavier normal
        nn.init.xavier_normal_(layer.weight)
        # Initialize the biases as zeros
        nn.init.zeros_(layer.bias)


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
# Training
# --------

# First, train the model for 2000 epochs, then export the result. This is like the burn-in
# period.
minimize_setting = dict(
    start_epoch=0, num_epochs=nepochs_initial, batch_size=bsize_list[0], lr=lr_list[0]
)
trained_model_file = MODEL_DIR / f"model_epoch{nepochs_initial}.pkl"

if trained_model_file.exists():
    model.load(trained_model_file)
else:
    print(f"Run initial training for {nepochs_initial} epochs")
    start_time = datetime.now()
    result = loss.minimize(method="Adam", **minimize_setting)
    end_time = datetime.now()
    print("Initial training time:", end_time - start_time)

# After that, we continue training for the specified total number of epochs, but we also
# export the model every 10 epochs.
ii = 0
ilr = 0  # Index to get the learning rate
nepochs_done = nepochs_initial
while nepochs_done < nepochs_total:
    start_epoch = nepochs_initial + ii * nepochs_save_period
    num_epochs = nepochs_save_period
    nepochs_done = start_epoch + num_epochs
    minimize_setting.update({"start_epoch": start_epoch, "num_epochs": num_epochs})

    if start_epoch > nepochs_list[ilr]:
        ilr += 1
        minimize_setting.update({"lr": lr_list[ilr], "batch_size": bsize_list[ilr]})

    trained_model_file = MODEL_DIR / f"model_epoch{nepochs_done}.pkl"
    if trained_model_file.exists():
        model.load(trained_model_file)
    else:
        start_time = datetime.now()
        result = loss.minimize(method="Adam", **minimize_setting)
        end_time = datetime.now()
        print(f"Training time up to epochs {nepochs_done}: {end_time - start_time}")

    ii += 1
    print(start_epoch, nepochs_done)
