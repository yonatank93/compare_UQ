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
import shutil

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

# Command line arguments
argv = sys.argv
if len(argv) == 3:
    # Argument contains config id and suffix given
    config_id = argv[1]
    loss_suffix = argv[2]
elif len(argv) == 2:
    # Argument only contains config id
    config_id = argv[1]
    loss_suffix = ""

# Directories
WORK_DIR = Path(__file__).absolute().parent
MODEL_DIR = WORK_DIR / "models" / f"initial_training{loss_suffix}"
FP_DIR = WORK_DIR / "fingerprints"
RES_DIR = WORK_DIR / "results"

# Architecture
Nlayers = 4  # Number of layers, excluding input layer, including outpt layer
Nnodes = 128  # Number of nodes per hidden layer
dropout_ratio = 0.1

# Optimizer settings
learning_rate = 0.001
batch_size = 100
nepochs_total = 50_000  # How many epochs to run in total
nepochs_burnin = 2000  # Run this many epochs first
nepochs_save_period = 10  # Then run and save every this many epochs


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
# model.set_save_metadata(
#     prefix=META_DIR / f"kliff_train_saved_model_{Nlayers}-{Nnodes}-{dropout_ratio}",
#     start=2000,
#     frequency=10,
# )

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
if config_id == "train":
    dataset_path = WORK_DIR / "carbon_training_set"
elif config_id == "test":
    dataset_path = WORK_DIR / "carbon_test_set"
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
    fingerprints_filename=FP_DIR / f"fingerprints_{config_id}.pkl",
    fingerprints_mean_stdev_filename=FP_DIR
    / f"fingerprints_{config_id}_mean_and_stdev.pkl",
)
loader = calc.get_compute_arguments(batch_size)


##########################################################################################
# Loss function
# -------------

residual_data = {"normalize_by_natoms": True}
loss = Loss(calc, residual_data=residual_data)

##########################################################################################
# Training
# --------

loss_values_file = RES_DIR / "loss" / f"loss_values_{config_id}{loss_suffix}.txt"
if loss_values_file.exists():
    loss_values = np.loadtxt(loss_values_file)
else:
    # First, train the model for 2000 epochs, then export the result. This is like the burn-in
    # period.
    minimize_setting = dict(
        start_epoch=0, num_epochs=1, batch_size=batch_size, lr=learning_rate
    )
    suffix = f"epochs{nepochs_burnin}"
    trained_model_file = MODEL_DIR / f"final_model_dropout_{suffix}.pkl"

    model.load(trained_model_file)
    loss_values = [nepochs_burnin, loss._get_loss_epoch(loader)]

    # After that, we continue training for the specified total number of epochs, but we also
    # export the model every 10 epochs.
    ii = 0
    nepochs_done = nepochs_burnin
    while nepochs_done < nepochs_total:
        try:
            start_epoch = nepochs_burnin + ii * nepochs_save_period + 1
            num_epochs = nepochs_save_period - 1
            nepochs_done = start_epoch + num_epochs
            minimize_setting.update({"start_epoch": start_epoch, "num_epochs": 1})

            suffix = f"epochs{nepochs_done}"
            trained_model_file = MODEL_DIR / f"final_model_dropout_{suffix}.pkl"

            model.load(trained_model_file)
            loss_values = np.row_stack(
                (loss_values, [nepochs_done, loss._get_loss_epoch(loader)])
            )
            ii += 1
            print(start_epoch, nepochs_done)
        except Exception as e:
            print(e)
            break

        np.savetxt(loss_values_file, loss_values)


# Write KIM model
idx = np.argmin(loss_values[:, 1])
best_epoch = int(loss_values[idx, 0])
model.load(MODEL_DIR / f"final_model_dropout_epochs{best_epoch}.pkl")
model.write_kim_model(f"DUNN_best_{config_id}")

# Move the best model pickle file
shutil.copy(
    MODEL_DIR / f"final_model_dropout_epochs{best_epoch}.pkl",
    WORK_DIR / f"model_best_{config_id}.pkl",
)
