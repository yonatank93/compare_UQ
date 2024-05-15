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
import json

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


# Read setting file
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
DATA_DIR = ROOT_DIR / "data"
with open(ROOT_DIR / "settings.json", "r") as f:
    settings = json.load(f)
partition = settings["partition"]

# Architecture
Nlayers = settings["Nlayers"]  # Number of layers, excluding input, including output
Nnodes = settings["Nnodes"]  # Number of nodes for each hidden layer
dropout_ratio = 0.1

# Optimizer settings
learning_rate = 1e-3
batch_size = 100
nepochs_total = 40_000  # How many epochs to run in total
nepochs_initial = 2000  # Run this many epochs first before start saving the model
nepochs_save_period = 10  # Then run and save every this many epochs
epoch_change_lr = 5000

# Directories
suffix = "_".join([str(n) for n in Nnodes])
PART_DIR = DATA_DIR / f"{partition}_partition_data"
FP_DIR = PART_DIR / "fingerprints"
RES_DIR = WORK_DIR / "results" / "training" / f"{partition}_partition_{suffix}"
if not RES_DIR.exists():
    print("Creating", RES_DIR)
    RES_DIR.mkdir()
META_DIR = RES_DIR / "metadata"
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
    prefix=META_DIR, start=nepochs_initial, frequency=nepochs_save_period
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

##########################################################################################
# Training
# --------

# First, train the model for 2000 epochs, then export the result. This is like the burn-in
# period.
minimize_setting = dict(
    start_epoch=0, num_epochs=nepochs_initial, batch_size=batch_size, lr=learning_rate
)
suffix = f"epochs{nepochs_initial}"
trained_model_file = MODEL_DIR / f"final_model_dropout_{suffix}.pkl"

if trained_model_file.exists():
    model.load(trained_model_file)
else:
    print(f"Run initial training for {nepochs_initial} epochs")
    start_time = datetime.now()
    result = loss.minimize(method="Adam", **minimize_setting)
    end_time = datetime.now()
    model.save(trained_model_file)
    print("Initial training time:", end_time - start_time)

# After that, we continue training for the specified total number of epochs, but we also
# export the model every 10 epochs.
ii = 0
nepochs_done = nepochs_initial
while nepochs_done < nepochs_total:
    start_epoch = nepochs_initial + ii * nepochs_save_period + 1
    num_epochs = nepochs_save_period - 1
    nepochs_done = start_epoch + num_epochs
    minimize_setting.update({"start_epoch": start_epoch, "num_epochs": num_epochs})

    if start_epoch > epoch_change_lr:
        minimize_setting.update({"lr": 1e-4})
    else:
        minimize_setting.update({"lr": learning_rate})

    suffix = f"epochs{nepochs_done}"
    trained_model_file = MODEL_DIR / f"final_model_dropout_{suffix}.pkl"

    if trained_model_file.exists():
        model.load(trained_model_file)
    else:
        start_time = datetime.now()
        result = loss.minimize(method="Adam", **minimize_setting)
        end_time = datetime.now()
        model.save(trained_model_file)
        print(f"Training time up to epochs {nepochs_done}: {end_time - start_time}")

    ii += 1
    print(start_epoch, nepochs_done)
