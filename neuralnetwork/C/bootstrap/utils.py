from pathlib import Path
import itertools
import json
import re

import numpy as np

WORK_DIR = Path(__file__).absolute().parent


# Setup regular expression search pattern
# Numbering pattern
numeric_const_pattern = r"""
[-+]? # optional sign
(?:
    (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
    |
    (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
)
# followed by optional exponent part if desired
(?: [Ee] [+-]? \d+ ) ?
"""
rx = re.compile(numeric_const_pattern, re.VERBOSE)


# Load version guide
version_guide_file = WORK_DIR / "version_guide.json"
if version_guide_file.exists():
    with open(version_guide_file, "r") as f:
        version_guide = json.load(f)

else:
    # Iterables
    Nlayers_list = [2, 3, 4]  # Excluding input layer, including output layer
    Nnodes_list = [96, 128, 144, 164, 196]
    dropout_ratio_list = [0.1, 0.2, 0.3]
    learning_rate_list = [0.01, 0.001, 0.0001]
    iterables = itertools.product(
        Nlayers_list, Nnodes_list, dropout_ratio_list, learning_rate_list
    )

    # Generate version guide
    version_guide = {}
    for ii, item in enumerate(iterables):
        Nlayers, Nnodes, dropout_ratio, learning_rate = item
        version_guide.update(
            {
                f"v{ii:04d}": {
                    "Nlayers": Nlayers,
                    "Nnodes": Nnodes,
                    "dropout_ratio": dropout_ratio,
                    "learning_rate": learning_rate,
                },
            }
        )

    # Export
    with open(version_guide_file, "w") as f:
        json.dump(version_guide, f, indent=4)


def get_version(settings):
    """Retrieve the version for given settings of model and training. The settings contain
    the number of hidden layers and nodes per layer, drop out ratio, and learning rate.
    The function returns the version in the format `vxxxx`, where `x` can be any numbers
    from 0 to 9.
    """
    Nlayers, Nnodes, dropout_ratio, learning_rate = settings
    # Find the version given the settings
    values = {
        "Nlayers": Nlayers,
        "Nnodes": Nnodes,
        "dropout_ratio": dropout_ratio,
        "learning_rate": learning_rate,
    }
    # Index of keys
    key_idx = list(version_guide.values()).index(values)
    version = list(version_guide)[key_idx]

    return version


def read_param_file(filename):
    """Read DUNN parameter file. This will returns the weights and biases as a super
    long, concatenated vector as well as the shape of the weights and biases.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Parse
    start_recording = False
    header = False
    params = []  # To store the parameters as a vector
    params_shape = []  # To store the shhape of the weights and biases
    for line in lines:
        if "weight of hidden layer 1" in line:
            # After this line, the file contains weights and biases.
            start_recording = True

        if "weight" in line and "layer" in line:
            # This is the header of the weights
            header = True
        elif "bias" in line and "layer" in line:
            # This is theheader of the biases
            header = True
        else:
            header = False

        if start_recording:
            # Extract numbers
            values = np.array([float(val) for val in rx.findall(line)])
            if header:
                # Get the shape of weights and biases
                if "hidden" in line:
                    values = values[1:]
                params_shape.append([int(val) for val in values])
            else:
                # Get parameters
                params = np.append(params, values)

    return params, params_shape
