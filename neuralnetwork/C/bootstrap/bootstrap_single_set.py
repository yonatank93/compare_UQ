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

argv = sys.argv
set_idx = int(argv[1])

# Read setting file
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
DATA_DIR = ROOT_DIR / "data"
with open(ROOT_DIR / "settings.json", "r") as f:
    settings = json.load(f)
partition = settings["partition"]

# Directories
PART_DIR = DATA_DIR / f"{partition}_partition_data"
FP_DIR = PART_DIR / "fingerprints"
RES_DIR = WORK_DIR / "results" / f"{partition}_partition"
if not RES_DIR.exists():
    RES_DIR.mkdir()
MODEL_DIR = RES_DIR / "models"
if not MODEL_DIR.exists():
    MODEL_DIR.mkdir()

# Architecture
Nlayers = 4  # Number of layers, excluding input layer, including outpt layer
Nnodes = 128  # Number of nodes per hidden layer
dropout_ratio = 0.1

# Optimizer settings
learning_rate = 1e-3
batch_size = 100
nepochs_total = 40_000  # How many epochs to run in total
nepochs_initial = 2000  # Run this many epochs first
nepochs_save_period = 10  # Then run and save every this many epochs
epoch_change_lr = 5000  # This is the epoch when we change the learning rate


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
    hidden_layer_mappings.append(nn.Dropout(dropout_ratio))
    hidden_layer_mappings.append(nn.Linear(Nnodes, Nnodes))
    hidden_layer_mappings.append(nn.Tanh())

model.add_layers(
    # input layer
    nn.Linear(descriptor.get_size(), Nnodes),  # Mapping from input layer to the first
    nn.Tanh(),  # hidden layer
    # hidden layer(s)
    *hidden_layer_mappings,  # Mappings between hidden layers in the middle
    # hidden layer(s)
    nn.Dropout(dropout_ratio),  # Mapping from the last hidden layer to the output layer
    nn.Linear(Nnodes, 1),
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
dataset_path = PART_DIR / "carbon_training_set"
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

# Load the bootstrap configurations
BS = BootstrapNeuralNetworkModel(loss, seed=seed)
bootstrap_fingerprints = BS.load_bootstrap_compute_arguments(
    WORK_DIR / f"bootstrap_fingerprints_{partition}.json"
)
calc.set_fingerprints(BS.bootstrap_compute_arguments[set_idx])

##########################################################################################
# Training
# --------

# Check if there some training has previously partially done
numeric_const_pattern = r"""
    [-+]? # optional sign
    (?:
        (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
        |
        (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
    )
    # followed by optional exponent part if desired
    (?: [Ee] [+-]? \d+ ) ?
    """
rx = re.compile(numeric_const_pattern, re.VERBOSE)
# If some training has been done, MODEL_DIR might not be empty. It is for sure empty if no
# training was done previously.
model_fnames = os.listdir(MODEL_DIR)
if len(model_fnames) == 0:
    # Fresh training
    start_epoch = 0
else:
    epochs_done = [int(float(rx.findall(fname)[0])) for fname in model_fnames]
    last_epoch = max(epochs_done)
    start_epoch = last_epoch + 1

    # Update the parameters to the last result
    model_fname = MODEL_DIR / f"model_epoch{int(last_epoch)}.pkl"
    model.load(model_fname)

minimize_setting = dict(batch_size=batch_size)

# Run training
if start_epoch < epoch_change_lr:
    # First, train the model using lr=1e-3
    print(
        f"Run initial training for upto {epoch_change_lr} epochs "
        f"using learning rate {learning_rate:0.1e}"
    )
    minimize_setting = dict(
        start_epoch=start_epoch,
        num_epochs=epoch_change_lr,
        lr=learning_rate,
    )
    start_time = datetime.now()
    _ = loss.minimize(method="Adam", **minimize_setting)
    end_time = datetime.now()
    start_epoch = epoch_change_lr + 1
    print("Initial training time:", end_time - start_time)


if start_epoch < nepochs_total:
    # After that, we continue training using lr=1e-4
    num_epochs = nepochs_total - start_epoch
    learning_rate *= 0.1
    minimize_setting.update(
        dict(start_epoch=start_epoch, num_epochs=num_epochs, lr=learning_rate)
    )

    print(
        f"Run the second stage of training for {num_epochs} epochs "
        f"using learning rate {learning_rate:0.1e}"
    )

    start_time = datetime.now()
    _ = loss.minimize(method="Adam", **minimize_setting)
    end_time = datetime.now()
    start_epoch = epoch_change_lr + 1
    print("Initial training time:", end_time - start_time)

# Export the parameters from the last epoch
last_params = calc.get_opt_params()
np.save(RES_DIR / "last_params.npy", last_params)
