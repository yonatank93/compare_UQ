# Compute the FIM of the entire neural network model using autograd functionality in pytorch.

from pathlib import Path
import json
import pickle
from tqdm import tqdm
import sys


import numpy as np
import scipy
from scipy.optimize import least_squares
from sklearn.model_selection import train_test_split
import torch

from kliff import nn
from kliff.calculators import CalculatorTorch
from kliff.dataset import Dataset
from kliff.dataset.weight import Weight
from kliff.descriptors import SymmetryFunction
from kliff.loss import Loss
from kliff.models import NeuralNetwork

# CL arguments
argv = sys.argv

# Random seed
seed = int(argv[2])
np.random.seed(seed)
torch.manual_seed(seed)


# # Setup
# ## Variables
# Read setting file
WORK_DIR = Path().absolute()
ROOT_DIR = WORK_DIR.parent
DATA_DIR = ROOT_DIR / "data"
with open(ROOT_DIR / "settings.json", "r") as f:
    settings = json.load(f)
partition = settings["partition"]
suffix = "_".join([str(n) for n in settings["Nnodes"]])
PART_DIR = DATA_DIR / f"{partition}_partition_data"
FP_DIR = PART_DIR / "fingerprints"
RES_DIR = WORK_DIR / "results" / f"{partition}_partition_{suffix}"
if not RES_DIR.exists():
    RES_DIR.mkdir(parents=True)
JAC_DIR = RES_DIR / "Jacobian"
if not JAC_DIR.exists():
    JAC_DIR.mkdir(parents=True)


# ## Model
# Architecture
Nlayers = settings[
    "Nlayers"
]  # Number of layers, excluding input layer, including outpt layer
Nnodes = settings["Nnodes"]  # Number of nodes per hidden layer
dropout_ratio = 0.0  # Don't use dropout

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

# Load best model
orig_model_path = (
    ROOT_DIR
    / "training_dropout"
    / "results"
    / "training"
    / f"{partition}_partition_{suffix}"
    / "model_best_train.pkl"
)
model.load(orig_model_path)


# ## Test set and calculator
# test set
dataset_path = PART_DIR / "carbon_test_set"
weight = Weight(energy_weight=1.0, forces_weight=np.sqrt(0.1))
tset = Dataset(dataset_path, weight)
configs = tset.get_configs()
nconfigs = len(configs)
print("Total number of configurations:", nconfigs)

# calculator
gpu = False
calc = CalculatorTorch(model, gpu=gpu)
_ = calc.create(
    configs,
    nprocs=20,
    reuse=True,
    fingerprints_filename=FP_DIR / f"fingerprints_test.pkl",
    fingerprints_mean_stdev_filename=FP_DIR / f"fingerprints_test_mean_and_stdev.pkl",
)
bestfit_params = calc.get_opt_params()

# Loss function
residual_data = {"normalize_by_natoms": True}
loss = Loss(calc, residual_data=residual_data)


# # Real calculation
# Total number of parameters, i.e., weights and biases
nparams = calc.get_num_opt_params()

# Each element in the loader only contains 1 configuration
loader = calc.get_compute_arguments(1)
loader_list = list(loader)  # Convert to list to make it easier to index

# We split the test set into test and validation sets
indices = np.arange(nconfigs)
valset_ratio = float(argv[1])  # Percentage of validation set compared to the entire data
idx_test, idx_val = train_test_split(indices, test_size=valset_ratio, random_state=seed)
# Update JAC_DIR
JAC_DIR = JAC_DIR / f"test_validation_{int(valset_ratio*100)}%_seed{seed}"
if not JAC_DIR.exists():
    JAC_DIR.mkdir()

# Export
with open(JAC_DIR / "test_validation_idx_split.pkl", "wb") as f:
    pickle.dump({"test": idx_test, "validation": idx_val}, f)

# Require gradient for all layers
for layer in model.layers:
    layer.requires_grad_(True)

# ## Compute the derivative of energy
device = model.device


def compute_jacobian_energy_one_config(ii):
    # ii = 0  # This index the batch in the loader
    batch = loader_list[ii]
    # Get the descriptor (zeta) for each atom in the first configuration
    jj = 0  # This index the configuration in the batch
    fingerprint_config = batch[jj]
    zeta_config = fingerprint_config["zeta"]
    natoms = len(zeta_config)
    # Compute the energy
    atom_energies = model(zeta_config)  # Energy of each atom in config ii

    # Weights and scaling factor
    weight = fingerprint_config["configuration"].weight
    # Whether we want to scale the prediction(s) by the number of atoms
    norm = 1 / natoms if residual_data["normalize_by_natoms"] else 1.0
    # Overall scaling factor
    scale = weight.config_weight * weight.energy_weight * norm

    jac_eng_atom = np.empty((0, nparams))
    for a in range(natoms):
        # a = 0  # Index of atom in the configuration
        # Make sure the gradients are initially zero, i.e., reset the gradient values.
        # This is to prevent undesired bahavior if we call backward method multiple
        # times.
        for layer in model.layers:
            layer.zero_grad()
        # For the scalar tensor (hence the index 0), call backward method to compute
        # the derivative of the scalar wrt all weights and biases.
        atom_energies[a].retain_grad()
        atom_energies[a].backward(retain_graph=True)

        parameters_grad = []
        for layer in model.layers:
            for param in layer.parameters():
                # I think each value in .grad property is the derivative of output (in
                # this case the energy) wrt the weights or biases. I need to confirm
                # this, but at least the shapes of these tensors agree with the claim.
                parameters_grad = np.append(parameters_grad, param.grad.detach().numpy())
        jac_eng_atom = np.row_stack((jac_eng_atom, parameters_grad))

    # So far, we get the derivative of energy for each atom. Now, we need to sum them
    # to get the derivative of configuration energy
    jac_eng_config = np.sum(jac_eng_atom, axis=0)
    # Apply the weights
    jac_eng_config *= scale
    return jac_eng_config


jac_eng_test_file = JAC_DIR / "jacobian_entire_network_energy_test.npy"
jac_eng_val_file = JAC_DIR / "jacobian_entire_network_energy_validation.npy"
if jac_eng_test_file.exists() and jac_eng_val_file.exists():
    print("Loading energy jacobian...")
    jac_eng_test = np.load(jac_eng_test_file)
    jac_eng_val = np.load(jac_eng_val_file)
else:
    # Compute Jacobian for energy predictions
    print("Compute energy jacobian...")
    jac_eng_test = np.array(
        list(
            tqdm(
                map(compute_jacobian_energy_one_config, idx_test),
                total=len(idx_test),
            )
        )
    )
    jac_eng_val = np.array(
        list(
            tqdm(
                map(compute_jacobian_energy_one_config, idx_val),
                total=len(idx_val),
            )
        )
    )
    # Export this Jacobian
    np.save(jac_eng_test_file, jac_eng_test)
    np.save(jac_eng_val_file, jac_eng_val)


# ## Compute the derivative of forces


def get_tensor_info(tensor):
    info = []
    for name in ["requires_grad", "is_leaf", "retains_grad", "grad_fn", "grad"]:
        info.append(f"{name}({getattr(tensor, name, None)})")
    return " ".join(info)


# Compute the derivative using autograd capability in pytorch
def compute_jacobian_forces_one_config(ii):
    # for batch in loader_list:  # Iterate over the configurations
    # ii = 0  # This index the batch in the loader
    batch = loader_list[ii]
    # Get the descriptor (zeta) for each atom in the first configuration
    jj = 0  # This index the configuration in the batch
    fingerprint_config = batch[jj]
    zeta_config = fingerprint_config["zeta"]
    zeta_config.requires_grad_(True)
    atom_energies = model.forward(zeta_config)  # Energy of each atom in config ii
    natoms = len(atom_energies)

    # First, compute the forces
    # Derivative of energy wrt descriptor
    dedzeta = torch.autograd.grad(atom_energies.sum(), zeta_config, create_graph=True)[0]
    zeta_config.requires_grad_(False)  # Don't need the gradient anymore
    # Derivative of descriptor wrt atomic coordinates
    dzetadr_forces = batch[0]["dzetadr_forces"]
    # Compute the forces by taking the dot product between dedzeta and dzetadr
    # We will take the derivative of this quantity.
    forces = -torch.tensordot(dedzeta, dzetadr_forces, dims=([0, 1], [0, 1]))

    # Before computing the derivative, we need to take care of the weights and scaling
    # factor
    weight = fingerprint_config["configuration"].weight
    config_weight = weight.config_weight
    forces_weight = weight.forces_weight
    # The forces weight can be a float or an array that applies to each element. We
    # will unify the format
    if isinstance(forces_weight, (float, int)):
        forces_weight = np.ones(natoms * 3) * forces_weight
    # Whether we want to scale the prediction(s) by the number of atoms
    norm = 1 / natoms if residual_data["normalize_by_natoms"] else 1.0
    # Overall scaling factor
    scale = config_weight * forces_weight * norm
    # Reshape the scale so we can use array multiplication
    scale = scale.reshape((-1, 1))

    # Compute the derivative
    grad_config = np.empty((0, nparams))
    for ielem in range(natoms * 3):  # Iterate over the forces element for each atom
        # Take derivative of forces wrt weights and biases
        for layer in model.layers:
            layer.zero_grad()
        forces[ielem].retain_grad()
        forces[ielem].backward(retain_graph=True)

        # Retrieve gradient values for each parameter
        grad_params = []
        for layer in model.layers:
            for param in layer.parameters():
                if param.grad is None:
                    grad = 0.0
                    # print(layer, param, get_tensor_info(param))
                else:
                    grad = param.grad.detach().numpy()
                grad_params = np.append(grad_params, grad)
        # Append to the gradient of the configuration
        grad_config = np.row_stack((grad_config, grad_params))
    # Multiply the derivative by the scaling factor
    grad_config *= scale

    return grad_config


jac_for_test_file = JAC_DIR / "jacobian_entire_network_forces_test.npy"
jac_for_val_file = JAC_DIR / "jacobian_entire_network_forces_validation.npy"
if jac_for_test_file.exists() and jac_for_val_file.exists():
    print("Loading forces jacobian...")
    jac_for_test = np.load(jac_for_test_file)
    jac_for_val = np.load(jac_for_val_file)
else:
    # Compute Jacobian for energy predictions
    print("Compute forces jacobian...")
    jac_for_test_ragged = list(
        tqdm(
            map(compute_jacobian_forces_one_config, idx_test),
            total=len(idx_test),
        )
    )
    # Stack the Jacobian
    jac_for_test = np.empty((0, nparams))
    for jac in tqdm(jac_for_test_ragged):
        jac_for_test = np.row_stack((jac_for_test, jac))

    jac_for_val_ragged = list(
        tqdm(
            map(compute_jacobian_forces_one_config, idx_val),
            total=len(idx_val),
        )
    )
    # Stack the Jacobian
    jac_for_val = np.empty((0, nparams))
    for jac in tqdm(jac_for_val_ragged):
        jac_for_val = np.row_stack((jac_for_val, jac))
    # Export this Jacobian
    np.save(jac_for_test_file, jac_for_test)
    np.save(jac_for_val_file, jac_for_val)


# Combine the Jacobian
# The derivative of energy will be on top of the derivative of forces
jac_test_file = JAC_DIR / "jacobian_entire_network_test.npy"
jac_val_file = JAC_DIR / "jacobian_entire_network_validation.npy"
if jac_test_file.exists() and jac_val_file.exists():
    print("Loading combined jacobian...")
    jac_test = np.load(jac_test_file)
    jac_val = np.load(jac_val_file)
else:
    print("Compute combined jacobian...")
    jac_test = np.row_stack((jac_eng_test, jac_for_test))
    print("Jacobian test set shape:", jac_test.shape)
    jac_val = np.row_stack((jac_eng_val, jac_for_val))
    print("Jacobian validation set shape:", jac_val.shape)
    np.save(jac_test_file, jac_test)
    np.save(jac_val_file, jac_val)

# Compute the FIM
# The FIM should have (much) smaller file size
fim_test_file = JAC_DIR / "fim_entire_network_test.npy"
fim_val_file = JAC_DIR / "fim_entire_network_validation.npy"
if fim_test_file.exists() and fim_val_file.exists():
    print("Loading fim...")
    fim_test = np.load(fim_test_file)
    fim_val = np.load(fim_val_file)
else:
    print("Compute fim...")
    fim_test = jac_test.T @ jac_test
    np.save(fim_test_file, fim_test)
    fim_val = jac_val.T @ jac_val
    np.save(fim_val_file, fim_val)

# Eigenvalue decomposition of the FIM
eighvecs_test_file = JAC_DIR / "eighvecs_entire_network_test.npy"
eighvals_test_file = JAC_DIR / "eighvals_entire_network_test.npy"
if eighvecs_test_file.exists() and eighvals_test_file.exists():
    print("Loading eigenvalues and eigenvectors from test set...")
    eighvals_test = np.load(eighvals_test_file)
    eighvecs_test = np.load(eighvecs_test_file)
else:
    # This calculation might take around 3 hours
    print("Compute eigenvalues and eigenvectors...")
    eighvals_test, eighvecs_test = scipy.linalg.eigh(fim_test)
    np.save(eighvals_test_file, eighvals_test)
    np.save(eighvecs_test_file, eighvecs_test)

eighvecs_val_file = JAC_DIR / "eighvecs_entire_network_validation.npy"
eighvals_val_file = JAC_DIR / "eighvals_entire_network_validation.npy"
if eighvecs_val_file.exists() and eighvals_val_file.exists():
    print("Loading eigenvalues and eigenvectors from validation set...")
    eighvals_val = np.load(eighvals_val_file)
    eighvecs_val = np.load(eighvecs_val_file)
else:
    # This calculation might take around 3 hours
    print("Compute eigenvalues and eigenvectors...")
    eighvals_val, eighvecs_val = scipy.linalg.eigh(fim_val)
    np.save(eighvals_val_file, eighvals_val)
    np.save(eighvecs_val_file, eighvecs_val)


# Estimate the effective number of parameters
# Test set
print("Estimate the number of effective parameters from the test set...")
loader_list_test = [loader_list[ii] for ii in idx_test]
C0_times_2_test = loss._get_loss_epoch(loader_list_test)


def find_optimal_N_objective(N):
    T = C0_times_2_test / N
    n = sum(eighvals_test > T)
    return N - n


opt_test = least_squares(find_optimal_N_objective, nparams / 2)
opt_N_test = int(np.ceil(opt_test.x))

# Validation set
print("Estimate the number of effective parameters from the validation set...")
loader_list_val = [loader_list[ii] for ii in idx_val]
C0_times_2_val = loss._get_loss_epoch(loader_list_val)


def find_optimal_N_objective(N):
    T = C0_times_2_val / N
    n = sum(eighvals_val > T)
    return N - n


opt_val = least_squares(find_optimal_N_objective, nparams / 2)
opt_N_val = int(np.ceil(opt_val.x))

# Export
print("Exporting...")
opt_N_dict = {
    "test": {"min_cost": C0_times_2_test, "ndata": len(jac_test), "N": opt_N_test},
    "validation": {"min_cost": C0_times_2_val, "ndata": len(jac_val), "N": opt_N_val},
}
with open(JAC_DIR / "number_effective_params.json", "w") as f:
    json.dump(opt_N_dict, f, indent=4)
