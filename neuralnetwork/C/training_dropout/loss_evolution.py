"""Look at how the loss evolves with the number of epochs."""

from pathlib import Path
import re
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from kliff.models import KIMModel
from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.dataset.weight import Weight
from kliff.loss import Loss


WORK_DIR = Path(__file__).absolute().parent
SFILE_DIR = WORK_DIR / "slurm_files"
RES_DIR = WORK_DIR / "results" / "loss"

# Setup regular expression search numbers
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


# Read slurm file
slurm_file_target = RES_DIR / "train_loss.out"
with open(slurm_file_target, "r") as f:
    slurm_file = f.readlines()

# Extract epoch and loss values
keywords = ["Epoch", "loss"]  # The line of interest contains these words

epoch_loss = np.empty((0, 2))
for line in tqdm(slurm_file):
    if all([word in line for word in keywords]):
        # The line contains information of the epoch and loss value
        # Extract the numbers
        numbers = rx.findall(line)
        epoch = int(numbers[0])
        loss = float(numbers[1])
        # Store
        epoch_loss = np.row_stack((epoch_loss, np.array([epoch, loss])))

# Plot loss vs epochs
plt.figure()
plt.plot(*(epoch_loss.T))
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.xlim(right=6000)
plt.show()
