"""This notebook is used write and install kim models from the results of uncertainty
from random initialization. I took the model correspondng to the last epoch (epoch 40,000)
of the training. Note that when I evaluate the target QoI later, I don't want to apply any
dropout and just want to use fully connected model.
"""

##########################################################################################
from pathlib import Path
import json
import argparse
import subprocess
from multiprocessing import Pool

import numpy as np
import torch

from kliff import nn
from kliff.calculators import CalculatorTorch
from kliff.dataset import Dataset
from kliff.descriptors import SymmetryFunction
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
    "-p", "--path", default=SETTINGS_DIR / "settings0.json", dest="settings_path"
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
tset = Dataset(dataset_path)
configs = tset.get_configs()

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
# Write KIM model
# ---------------
def install_model_given_idx(set_idx):
    SAMPLE_DIR = RES_DIR / f"{set_idx:03d}"

    # Load last parameters
    last_param_file = SAMPLE_DIR / "last_params.npy"
    if last_param_file.exists():
        last_params = np.load(last_param_file)
        calc.update_model_params(last_params)
        # Write model
        modelname = f"DUNN_C_randinit_{set_idx:03d}"
        model.write_kim_model(SAMPLE_DIR / modelname)

        # Install
        subprocess.run(
            [
                "kim-api-collections-management",
                "install",
                "--force",
                "user",
                SAMPLE_DIR / modelname,
            ]
        )


with Pool(25) as p:
    p.map(install_model_given_idx, range(100))
