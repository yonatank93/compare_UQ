"""This notebook is used write and install kim models from the results of bootstrapping.
I took the model correspondng to the last epoch (epoch 40,000) of the training. Note that
when I evaluate the target QoI later, I don't want to apply any dropout and just want to
use fully connected model.
"""

##########################################################################################
from pathlib import Path
import json
import sys
import subprocess

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


argv = sys.argv
if len(argv) > 1:
    # Mode argument is speciied
    mode = argv[1]
else:
    mode = "install"


##########################################################################################
# Initial Setup
# -------------

# Read setting file
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
with open(ROOT_DIR / "settings.json", "r") as f:
    settings = json.load(f)
partition = settings["partition"]

# Directories
TRAIN_MODEL_DIR = (
    ROOT_DIR
    / "training_dropout"
    / "results"
    / "training"
    / f"{partition}_partition"
    / "models"
)
PART_DIR = ROOT_DIR / f"{partition}_partition_data"
FP_DIR = PART_DIR / "fingerprints"
RES_DIR = WORK_DIR / "results" / f"{partition}_partition"
if not RES_DIR.exists():
    RES_DIR.mkdir(parents=True)

# Architecture
Nlayers = 4  # Number of layers, excluding input layer, including outpt layer
Nnodes = 128  # Number of nodes per hidden layer
dropout_ratio = 0.1


##########################################################################################
# Models
# -----

# Descriptor
descriptor = SymmetryFunction(
    cut_name="cos", cut_dists={"C-C": 5.0}, hyperparams="set51", normalize=True
)

# With dropout. This is so that we can load the model file exported during the training.
model1 = NeuralNetwork(descriptor)
# Without dropout, which is the model that we will use in the UQ.
model2 = NeuralNetwork(descriptor)


# Layers
hidden_layer_mappings1 = []
hidden_layer_mappings2 = []
for _ in range(Nlayers - 2):
    hidden_layer_mappings1.append(nn.Dropout(dropout_ratio))
    hidden_layer_mappings1.append(nn.Linear(Nnodes, Nnodes))
    hidden_layer_mappings2.append(nn.Linear(Nnodes, Nnodes))
    hidden_layer_mappings1.append(nn.Tanh())
    hidden_layer_mappings2.append(nn.Tanh())

model1.add_layers(
    # input layer
    nn.Linear(descriptor.get_size(), Nnodes),  # Mapping from input layer to the first
    nn.Tanh(),  # hidden layer
    # hidden layer(s)
    *hidden_layer_mappings1,  # Mappings between hidden layers in the middle
    # hidden layer(s)
    nn.Dropout(dropout_ratio),  # Mapping from the last hidden layer to the output layer
    nn.Linear(Nnodes, 1),
    # output layer
)
model2.add_layers(
    # input layer
    nn.Linear(descriptor.get_size(), Nnodes),  # Mapping from input layer to the first
    nn.Tanh(),  # hidden layer
    # hidden layer(s)
    *hidden_layer_mappings2,  # Mappings between hidden layers in the middle
    # hidden layer(s)
    nn.Linear(Nnodes, 1),
    # output layer
)


##########################################################################################
# Training set and calculator
# ---------------------------

# training set
dataset_path = PART_DIR / "carbon_training_set"
tset = Dataset(dataset_path)
configs = tset.get_configs()

# calculator
gpu = False
calc1 = CalculatorTorch(model1, gpu=gpu)
calc2 = CalculatorTorch(model2, gpu=gpu)
_ = calc1.create(
    configs,
    nprocs=20,
    reuse=True,
    fingerprints_filename=FP_DIR / f"fingerprints_train.pkl",
    fingerprints_mean_stdev_filename=FP_DIR / f"fingerprints_train_mean_and_stdev.pkl",
)
_ = calc2.create(
    configs,
    nprocs=20,
    reuse=True,
    fingerprints_filename=FP_DIR / f"fingerprints_train.pkl",
    fingerprints_mean_stdev_filename=FP_DIR / f"fingerprints_train_mean_and_stdev.pkl",
)


##########################################################################################
# Write KIM model
# ---------------
for set_idx, epoch in enumerate(np.arange(30000, 40000, 100)):
    SAMPLE_DIR = RES_DIR / f"{set_idx:03d}"

    # Load last parameters
    model_file = TRAIN_MODEL_DIR / f"final_model_dropout_epochs{epoch}.pkl"
    if model_file.exists():
        # Load the model and get the parameters. We will use these parameters to load the
        # parameters into the second model without dropout.
        model1.load(model_file)
        params = calc1.get_opt_params()
        calc2.update_model_params(params)
        # Export the parameters
        # Write model
        modelname = f"DUNN_C_losstraj_{set_idx:03d}"
        model2.write_kim_model(SAMPLE_DIR / modelname)

        if mode == "install":
            # Install
            subprocess.run(
                [
                    "kim-api-collections-management",
                    "install",
                    "user",
                    SAMPLE_DIR / modelname,
                ]
            )
        elif mode == "uninstall":
            # Uninstall
            subprocess.run(
                ["kim-api-collections-management", "remove", "--force", modelname]
            )
    else:
        continue
