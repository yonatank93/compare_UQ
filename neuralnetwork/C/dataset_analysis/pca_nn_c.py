"""In this script, I want to look at the similarity between atomic environment by using
PCA.
The PCA mapping will be trained on the training dataset only.
"""

##########################################################################################
from pathlib import Path
from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
# Train PCA
# ---------
print("Training PCA...")

pca_matrices_file = RES_DIR / "pca_matrices.pkl"
if pca_matrices_file.exists():
    # Load U, S, and V
    with open(pca_matrices_file, "rb") as f:
        matrices = pickle.load(f)
    S = matrices["S"]
    V = matrices["V"]
else:
    print("Compute PCA...")
    # Note: by convention, the fingerprint data is already normalized.
    data_train = fp_data["training"]["values"].iloc[:, 1:]
    # We only need the singular values and the eigenvectors. If we just do SVD, it will
    # take too much memory and take too long. A work around is to do dot product and do
    # eigenvalue decomposition instead.
    data2 = data_train.T @ data_train
    # Eigenvalue decomposition
    L, V = np.linalg.eigh(data2)
    # Singular values are the square root of the eigenvalues
    # Some eigenvalues are negative due to numerical error. Also, we want a decending
    # order
    S = np.sqrt(np.abs(L))[::-1]
    # Decending eigenvectors
    V = V[:, ::-1]
    # Save
    with open(pca_matrices_file, "wb") as f:
        pickle.dump({"S": S, "V": V}, f)


##########################################################################################
# PCA embeddings
# ---------------
print("Compute fingerprints embeddings...")


def transform(data, npc=2):
    """Transform the data to the PCA space.

    Parameters
    ----------
    data: np.ndarray
        The data to transform.
    npc: int (optional)
        The number of principal components to keep.

    Returns
    -------
    np.ndarray
        The transformed data.
    """
    return data @ V[:, :npc]


# Compute
npc = 3  # Number of principle components
for key in fp_data.keys():
    print("Processing", key)
    fp_data[key]["embedding"] = transform(fp_data[key]["values"].iloc[:, 1:], npc=npc)
    # Save embeddings
    np.savetxt(
        RES_DIR / f"pca_embedding_{npc}d_{key}.txt",
        fp_data[key]["embedding"],
        delimiter=",",
    )


# # Plot
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection="3d")


# def plot_embedding(data_df, embedding, fig=None, ax=None, colors=None, **kwargs):
#     if fig is None:
#         fig = plt.figure()
#     if ax is None:
#         ax = fig.add_subplot(111, projection="3d")
#     if colors is None:
#         # Get colors by structure
#         colors = []
#         for ii in range(len(data_df)):
#             if "diamond" in data_df.iloc[ii, 0]:
#                 colors.append("tab:red")
#             elif "monolayer" in data_df.iloc[ii, 0]:
#                 colors.append("tab:blue")
#             elif "bilayer" in data_df.iloc[ii, 0]:
#                 colors.append("tab:orange")
#             elif "graphite" in data_df.iloc[ii, 0]:
#                 colors.append("tab:green")
#     # Plot
#     ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=colors, **kwargs)
