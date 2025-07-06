"""This notebook is used write and install kim models from the results of bootstrapping.
I took the model correspondng to the last epoch (epoch 40,000) of the training. Note that
when I evaluate the target QoI later, I don't want to apply any dropout and just want to
use fully connected model.
"""

##########################################################################################
from pathlib import Path
import json
import argparse
import subprocess
from multiprocessing import Pool

import numpy as np
import torch
from emcee.autocorr import integrated_time

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
# Training result directory
TRAIN_DIR = ROOT_DIR / "training" / "results" / settings_path.with_suffix("").name

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
for ii in range(Nlayers - 2):
    hidden_layer_mappings1.append(nn.Dropout(dropout_ratio))
    hidden_layer_mappings1.append(nn.Linear(Nnodes[ii], Nnodes[ii + 1]))
    hidden_layer_mappings2.append(nn.Linear(Nnodes[ii], Nnodes[ii + 1]))
    hidden_layer_mappings1.append(nn.Tanh())
    hidden_layer_mappings2.append(nn.Tanh())

model1.add_layers(
    # input layer
    nn.Linear(descriptor.get_size(), Nnodes[0]),  # Mapping from input layer to the first
    nn.Tanh(),  # hidden layer
    # hidden layer(s)
    *hidden_layer_mappings1,  # Mappings between hidden layers in the middle
    # hidden layer(s)
    nn.Dropout(dropout_ratio),  # Mapping from the last hidden layer to the output layer
    nn.Linear(Nnodes[-1], 1),
    # output layer
)
model2.add_layers(
    # input layer
    nn.Linear(descriptor.get_size(), Nnodes[0]),  # Mapping from input layer to the first
    nn.Tanh(),  # hidden layer
    # hidden layer(s)
    *hidden_layer_mappings2,  # Mappings between hidden layers in the middle
    # hidden layer(s)
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
calc1 = CalculatorTorch(model1, gpu=gpu)
calc2 = CalculatorTorch(model2, gpu=gpu)
_ = calc1.create(
    configs,
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
# Analyze autocorrelation
# -----------------------
# I think it is save to set to just use the last 10_000 epochs. If we look at the training
# cost, the cost has already plateaued after epoch 30_000.
burnin = nepochs_total - 10_000

# Load the training loss data
loss_train = np.loadtxt(TRAIN_DIR / "loss_values_train.txt")
idx_samples = np.where(loss_train[:, 0] >= burnin)[0]
loss_samples = loss_train[idx_samples, 1]

# Estimate the autocorrelation length
acorr_est = integrated_time(loss_samples, c=1, has_walkers=False, quiet=True)
print("Estimated autocorrelation length:", acorr_est)
# The estimated autocorrelation length is pretty low; it is less than 1.0. So, it is save
# to sample every 100 epochs.
sample_freq = 100


##########################################################################################
# Write and install KIM model
# ---------------------------
epoch_list = np.arange(burnin, nepochs_total, sample_freq) + sample_freq
# We added sample_freq so that in exclude burnin point (e.g., epoch 30,000) and include
# last point (e.g., epoch 40,000)


def install_models(set_idx):
    """Wrapper function to write and install KIM model given the set_idx."""
    epoch = epoch_list[set_idx]
    SAMPLE_DIR = RES_DIR / f"{set_idx:03d}"

    # Load last parameters
    model_file = TRAIN_DIR / "models" / f"model_epoch{epoch}.pkl"
    print(set_idx, model_file.exists())
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


# Run
with Pool(25) as p:
    p.map(install_models, range(len(epoch_list)))
