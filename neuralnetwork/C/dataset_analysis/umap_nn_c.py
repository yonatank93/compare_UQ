"""In this script, I want to look at the similarity between atomic environment by using
UMAP.
The UMAP mapping will be trained on the training dataset only.
"""

##########################################################################################
from pathlib import Path
from tqdm import tqdm
import joblib

import numpy as np
import pandas as pd
import torch

from kliff import nn
from kliff.calculators import CalculatorTorch
from kliff.dataset import Dataset
from kliff.descriptors import SymmetryFunction
from kliff.models import NeuralNetwork

import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Random seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_tensor_type(torch.DoubleTensor)
NDIM = 3


##########################################################################################
# Initial Setup
# -------------

# Read setting file
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RES_DIR = WORK_DIR / "results"

# Fingerprints csv information
fp_data = {
    "training": {"filename": RES_DIR / "fingerprints_training.csv"},
    "test": {"filename": RES_DIR / "fingerprints_test.csv"},
    "diamond_energy_curve": {
        "filename": RES_DIR / "fingerprints_diamond_energy_curve.csv"
    },
    "graphite_energy_curve": {
        "filename": RES_DIR / "fingerprints_graphite_energy_curve.csv"
    },
    "graphene_energy_curve": {
        "filename": RES_DIR / "fingerprints_graphene_energy_curve.csv"
    },
}

# Load
for key in fp_data.keys():
    fp_data[key]["values"] = pd.read_csv(fp_data[key]["filename"], header=0, index_col=0)


##########################################################################################
# UMAP
# ----

# Train UMAP
umap_reducer_file = RES_DIR / f"umap_reducer_{NDIM}d.sav"
if umap_reducer_file.exists():
    # Load the reducer
    print("Loading UMAP reducer...")
    reducer = joblib.load(umap_reducer_file)
else:
    print("Training UMAP reducer...")
    # Create reducer
    reducer = umap.UMAP(
        n_components=NDIM, n_jobs=20, verbose=True, learning_rate=0.1, n_epochs=5000
    )
    # Train
    data_train = df_train.iloc[:, 1:]
    reducer.fit(data_train)
    # Save
    joblib.dump(reducer, umap_reducer_file)


##########################################################################################
# UMAP embeddings
# ---------------

# Training set
print("Fingerprints embedding for the training set...")
embedding_train_file = RES_DIR / f"umap_embedding_{NDIM}d_train.npy"
if embedding_train_file.exists():
    embeddings_train = np.load(embedding_train_file)
else:
    data_train = df_train.iloc[:, 1:]
    embedding_train = reducer.transform(data_train)
    np.save(embedding_train_file, embedding_train)

# Test set
print("Fingerprints embedding for the test set...")
# Get the fingerprints dataframe
fp_file = RES_DIR / "fingerprints_test.csv"
if not fp_file.exists():
    print("Creating fingerprints dataframe...")
    df_test = get_fingerprints_dataframe(
        PART_DIR / "carbon_test_set",
        FP_DIR / "fingerprints_test.pkl",
        FP_DIR / "fingerprints_test_mean_and_stdev.pkl",
    )
    # Export to csv file
    df_test.to_csv(fp_file)
else:
    print("Reading fingerprints dataframe...")
    df_test = pd.read_csv(fp_file, header=0, index_col=0)
# Embedding for the test set
embedding_test_file = RES_DIR / f"umap_embedding_{NDIM}d_test.npy"
if embedding_test_file.exists():
    embedding_test = np.load(embedding_test_file)
else:
    data_test = df_test.iloc[:, 1:]
    embedding_test = reducer.transform(data_test)
    np.save(embedding_test_file, embedding_test)

# Other configurations --- Energy vs lattice parameter diamond
print("Fingerprints embedding for the other set...")
# Get the fingerprints dataframe --- Assume that the fingerprint dataframe has been
# constructed previously
fp_file = RES_DIR / "fingerprints_diamond_energy_curve.csv"
df_other = pd.read_csv(fp_file, header=0, index_col=0)
# Embedding for the other set
embedding_other_file = RES_DIR / f"umap_embedding_{NDIM}d_other.npy"
if embedding_other_file.exists():
    embedding_other = np.load(embedding_other_file)
else:
    data_other = df_other.iloc[:, 1:]
    embedding_other = reducer.transform(data_other)
    np.save(embedding_other_file, embedding_other)


# Plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")


def plot_embedding(data_df, embedding, fig=None, ax=None, colors=None, **kwargs):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ndim = embedding.shape[1]
        if ndim == 2:
            ax = fig.add_subplot(111)
        elif ndim == 3:
            ax = fig.add_subplot(111, projection="3d")
    if colors is None:
        # Get colors by structure
        colors = []
        for ii in range(len(data_df)):
            if "diamond" in data_df.iloc[ii, 0]:
                colors.append("tab:red")
            elif "monolayer" in data_df.iloc[ii, 0]:
                colors.append("tab:blue")
            elif "bilayer" in data_df.iloc[ii, 0]:
                colors.append("tab:orange")
            elif "graphite" in data_df.iloc[ii, 0]:
                colors.append("tab:green")
    # Plot
    ax.scatter(*(embedding.T), c=colors, **kwargs)
