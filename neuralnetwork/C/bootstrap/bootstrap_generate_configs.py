from pathlib import Path

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

# Directories
WORK_DIR = Path(__file__).absolute().parent
FP_DIR = WORK_DIR / "fingerprints"

# Architecture
Nlayers = 4  # Number of layers, excluding input layer, including outpt layer
Nnodes = 128  # Number of nodes per hidden layer
dropout_ratio = 0.1

# Optimizer settings
learning_rate = 0.001
batch_size = 100

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
dataset_path = WORK_DIR / "carbon_training_set"
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
bootstrap_fingerprints_file = Path("bootstrap_fingerprints.json")
if bootstrap_fingerprints_file.exists():
    BS.load_bootstrap_compute_arguments(bootstrap_fingerprints_file)
else:
    BS.generate_bootstrap_compute_arguments(nsamples_target)
    BS.save_bootstrap_compute_arguments(bootstrap_fingerprints_file)
