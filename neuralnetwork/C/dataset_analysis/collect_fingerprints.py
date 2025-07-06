"""I'm using this script to collect the fingerprints and put them into csv file. I will
collect the fingerprints of the training data, test data, as well as configurations
used when computing energy vs lattice parameters.
"""

##########################################################################################
from pathlib import Path
from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd

from kliff.dataset import Dataset
from kliff.descriptors import SymmetryFunction


##########################################################################################
# Initial Setup
# -------------

# Read setting file
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RES_DIR = WORK_DIR / "results"

# Directories
PART_DIR = DATA_DIR / "mingjian_partition_data"
FP_DIR = PART_DIR / "fingerprints"


##########################################################################################
# Descriptor
# ----------

# Descriptor
descriptor = SymmetryFunction(
    cut_name="cos", cut_dists={"C-C": 5.0}, hyperparams="set51", normalize=True
)
# Fingerprints mean and standard deviation for normalization
with open(FP_DIR / "fingerprints_train_mean_and_stdev.pkl", "rb") as f:
    fp_mean_stdev = pickle.load(f)
descriptor.mean = fp_mean_stdev["mean"]
descriptor.stdev = fp_mean_stdev["stdev"]


##########################################################################################
# Compute fingerprints
# --------------------
print("Preparing to compute fingerprints...")

# Put fingerprints in a dataframe
structures = ["diamond", "monolayer", "bilayer", "graphite", "graphene"]


# This function puts the configuration fingerprints into a dataframe
def get_fingerprints_dataframe(dataset_path):
    # Read the configurations
    dataset_path = dataset_path
    tset = Dataset(dataset_path)
    configs = tset.get_configs()

    # Compute the fingerprints
    df_array = np.empty((0, 52), dtype="<U32")  # 1 col for structure, 51 for descriptors
    for conf in tqdm(configs):
        # Number of atoms
        natoms = conf.get_num_atoms()
        # Get the structure type
        for struct in structures:
            if f"{dataset_path.name}/{struct}" in conf.identifier:
                structure = struct
                break
        # Compute fingerprints
        fp = descriptor.transform(conf)[0]
        fp = (fp - descriptor.mean) / descriptor.stdev  # Normalize
        # Prepare array to append
        df_fp = np.hstack((np.array([structure] * natoms).reshape((-1, 1)), fp))
        # Append
        df_array = np.vstack((df_array, df_fp))

    # Create a dataframe
    columns = np.append("structure", range(51))
    df = pd.DataFrame(df_array, columns=columns)
    return df


# File information
config_dirs = {
    "training": PART_DIR / "carbon_training_set",
    "test": PART_DIR / "carbon_test_set",
    "diamond_energy_curve": ROOT_DIR / "energyvslatconst/dft_data/xyz_files/diamond",
    "graphite_energy_curve": ROOT_DIR / "energyvslatconst/dft_data/xyz_files/graphite",
    "graphene_energy_curve": ROOT_DIR / "energyvslatconst/dft_data/xyz_files/graphene",
}

# Fingerprints calculations
for key, config_dir in config_dirs.items():
    print(f"Computing fingerprints for {key}...")
    df_file = RES_DIR / f"fingerprints_{key}.csv"
    if not df_file.exists():
        df = get_fingerprints_dataframe(config_dir)
        df.to_csv(df_file, index=False)
