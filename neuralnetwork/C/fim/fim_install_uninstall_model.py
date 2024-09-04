"""This script ins exclusively used to install or uninstall the model ensembles for this
method. The model ensembles and the KIM models need to be written prior to running this
script.
"""

##########################################################################################
from pathlib import Path
import json
import argparse
import subprocess
from multiprocessing import Pool


##########################################################################################
# Initial Setup
# -------------

# Read settings
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
DATA_DIR = ROOT_DIR / "data"

# settings = {"partition": "mingjian", "Nlayers": 4, "Nnodes": [128, 128, 128]}
arg_parser = argparse.ArgumentParser("Settings of the calculations")
arg_parser.add_argument("-p", "--partition", dest="partition")
arg_parser.add_argument("-l", "--nlayers", type=int, dest="nlayers")
arg_parser.add_argument("-n", "--nnodes", nargs="+", type=int, dest="nnodes")
arg_parser.add_argument("-m", "--mode", dest="mode")
args = arg_parser.parse_args()
# Read settings from file for missing arguments
with open(ROOT_DIR / "settings.json", "r") as f:
    settings_from_file = json.load(f)
# Missing values
if args.partition is None:
    args.partition = settings_from_file["partition"]
if args.nlayers is None:
    args.nlayers = settings_from_file["Nlayers"]
if args.nnodes is None:
    args.nnodes = settings_from_file["NNodes"]
# Construct settings dictionary
settings = {"partition": args.partition, "Nlayers": args.nlayers, "Nnodes": args.nnodes}


# Directories
partition = settings["partition"]
suffix = "_".join([str(n) for n in settings["Nnodes"]])
PART_DIR = DATA_DIR / f"{partition}_partition_data"
RES_DIR = WORK_DIR / "results" / f"{partition}_partition_{suffix}" / args.mode
if not RES_DIR.exists():
    RES_DIR.mkdir()


##########################################################################################
# Write KIM model
# ---------------


def install_models(set_idx):
    SAMPLE_DIR = RES_DIR / f"{set_idx:03d}"
    modelname = f"DUNN_C_fimbayes_{set_idx:03d}"
    modelpath = SAMPLE_DIR / modelname
    if modelpath.exists():
        # Uninstall first, then install
        subprocess.run(["kim-api-collections-management", "remove", "--force", modelname])
        subprocess.run(["kim-api-collections-management", "install", "user", modelpath])
    else:
        raise ValueError(f"{modelname} is not found. Please write this KIM model first.")


# Run
with Pool(50) as p:
    p.map(install_models, range(100))
