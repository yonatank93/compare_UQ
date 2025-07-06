"""This script is to test running MD simulation to compute the virial test for model
ensembles obtained from the SNAPSHOT ensemble.
"""

from pathlib import Path
import json
import re
import argparse
import sys

import numpy as np

WORK_DIR = Path(__file__).absolute().parent
sys.path.append(str(WORK_DIR.parent))

from virialstress import virialvslatconst

# Read settings
ROOT_DIR = WORK_DIR.parent
SETTINGS_DIR = ROOT_DIR / "settings"

# Command line argument
arg_parser = argparse.ArgumentParser("Settings file path")
arg_parser.add_argument(
    "-p", "--path", default=SETTINGS_DIR / "settings0.json", dest="settings_path"
)
arg_parser.add_argument("-i", "--set-idx", type=int, default=0, dest="set_idx")
args = arg_parser.parse_args()

settings_path = Path(args.settings_path)
with open(settings_path, "r") as f:
    settings = json.load(f)
RES_DIR = WORK_DIR / "results" / re.match(r"^[^_\.]+", settings_path.name).group()


# Run stress calculation
alist = np.linspace(0.95, 1.05, 11) * 2.466
idx = args.set_idx
modelname = f"DUNN_C_losstraj_{idx:03d}"
path = RES_DIR / f"{idx:03d}" / "virial_stress_graphene"

stress = virialvslatconst(modelname, alist, None, path)
