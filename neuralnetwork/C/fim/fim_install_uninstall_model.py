"""This script ins exclusively used to install or uninstall the model ensembles for this
method. The model ensembles and the KIM models need to be written prior to running this
script.
"""

##########################################################################################
from pathlib import Path
import json
import sys
import subprocess


argv = sys.argv
if len(argv) > 1:
    # Mode argument is speciied
    mode = argv[1]
else:
    mode = "install"


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

# Directories
suffix = "_".join([str(n) for n in settings["Nnodes"]])
PART_DIR = DATA_DIR / f"{partition}_partition_data"
RES_DIR = WORK_DIR / "results" / f"{partition}_partition_{suffix}"
if not RES_DIR.exists():
    RES_DIR.mkdir()


##########################################################################################
# Write KIM model
# ---------------
for set_idx in range(100):
    SAMPLE_DIR = RES_DIR / f"{set_idx:03d}"
    modelname = f"DUNN_C_fimbayes_{set_idx:03d}"
    modelpath = SAMPLE_DIR / modelname
    if not modelpath.exists():
        print(f"{modelname} is not found. Please write this KIM model first.")
        continue

    if mode == "install":
        # Install
        subprocess.run(["kim-api-collections-management", "install", "user", modelpath])
    elif mode == "uninstall":
        # Uninstall
        subprocess.run(["kim-api-collections-management", "remove", "--force", modelname])
