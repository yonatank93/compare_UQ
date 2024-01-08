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
PART_DIR = ROOT_DIR / f"{partition}_partition_data"
FP_DIR = PART_DIR / "fingerprints"
RES_DIR = WORK_DIR / "results" / f"{partition}_partition"
if not RES_DIR.exists():
    RES_DIR.mkdir()

# Architecture
Nlayers = 4  # Number of layers, excluding input layer, including outpt layer
Nnodes = 128  # Number of nodes per hidden layer


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
for _ in range(Nlayers - 2):
    hidden_layer_mappings.append(nn.Linear(Nnodes, Nnodes))
    hidden_layer_mappings.append(nn.Tanh())

model.add_layers(
    # input layer
    nn.Linear(descriptor.get_size(), Nnodes),  # Mapping from input layer to the first
    nn.Tanh(),  # hidden layer
    # hidden layer(s)
    *hidden_layer_mappings,  # Mappings between hidden layers in the middle
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
for set_idx in range(100):
    SAMPLE_DIR = RES_DIR / f"{set_idx:03d}"

    # Load last parameters
    last_param_file = SAMPLE_DIR / "last_params.npy"
    if last_param_file.exists():
        last_params = np.load(last_param_file)
        calc.update_model_params(last_params)
        # Write model
        modelname = f"DUNN_C_bootstrap_{set_idx:03d}"
        model.write_kim_model(SAMPLE_DIR / modelname)

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
