"""Split the configurations into training set and test set."""

##########################################################################################
from pathlib import Path
import json
import shutil
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split
from kliff.dataset import Dataset

np.random.seed(1)

WORK_DIR = Path(__file__).absolute().parent


##########################################################################################
# Training set and calculator
# ---------------------------

# Read dataset
print("Reading dataset")
dataset_path = WORK_DIR / "carbon_energies_forces"
tset = Dataset(dataset_path)
configs = tset.get_configs()

# Split
print("Train-test split")
train_size = 0.9
train_configs, test_configs = train_test_split(configs, train_size=train_size)

train = {}
test = {}

for conf in train_configs:
    identifier = conf.identifier
    path = Path(identifier)
    name = path.name
    structure = path.parent.name
    if structure not in train:
        train.update({structure: []})
    train[structure].append(identifier)

for conf in test_configs:
    identifier = conf.identifier
    path = Path(identifier)
    name = path.name
    structure = path.parent.name
    if structure not in test:
        test.update({structure: []})
    test[structure].append(identifier)

configs_dict = {"train": train, "test": test}
json.dump(configs_dict, open("configs_train_test.json", "w"), indent=4)


# Put the train and test configurations into separate folders
train_configs_dir = WORK_DIR / "carbon_training_set"
test_configs_dir = WORK_DIR / "carbon_test_set"

# Clean up
if train_configs_dir.is_dir():
    shutil.rmtree(train_configs_dir)
if test_configs_dir.is_dir():
    shutil.rmtree(test_configs_dir)

# Copy training configurations
print("Training configurations")
train_configs_dir.mkdir()
train_configs_dict = configs_dict["train"]
for structure, paths in train_configs_dict.items():
    print(f"Copying {structure} configurations")
    structure_dir = train_configs_dir / structure
    structure_dir.mkdir()
    for path in tqdm(paths):
        shutil.copy(path, structure_dir)

# Copy test configurations
print("Test configurations")
test_configs_dir.mkdir()
test_configs_dict = configs_dict["test"]
for structure, paths in test_configs_dict.items():
    print(f"Copying {structure} configurations")
    structure_dir = test_configs_dir / structure
    structure_dir.mkdir()
    for path in tqdm(paths):
        shutil.copy(path, structure_dir)
