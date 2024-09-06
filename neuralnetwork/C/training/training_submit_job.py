"""Submit job to train the NN model."""

from pathlib import Path
import subprocess
import itertools
import copy
import json

WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
SETTINGS_DIR = ROOT_DIR / "settings"

# Iterables - Generate a list of settings
# Dataset
dataset_path = [
    "/home/yonatank/nobackup/autodelete/compare_UQ/neuralnetwork/C/data/mingjian_partition_data/carbon_training_set",
    # "/home/yonatank/nobackup/autodelete/compare_UQ/neuralnetwork/C/data/yonatan_partition_data/carbon_training_set",
]
fingerprints_path = [
    "/home/yonatank/nobackup/autodelete/compare_UQ/neuralnetwork/C/data/mingjian_partition_data/fingerprints",
    # "/home/yonatank/nobackup/autodelete/compare_UQ/neuralnetwork/C/data/yonatan_partition_data/fingerprints",
]

# Architecture
Nlayers_try = [4]  # Excluding input layer, including output layer
Nnodes_try = [128]  # [64, 128,
Nnodes_list = []
for Nlayers in Nlayers_try:
    Nnodes_list.extend(list(itertools.product(Nnodes_try, repeat=Nlayers - 1)))
Nlayers_list = [len(n) + 1 for n in Nnodes_list]
dropout_list = [0.1]  # [0.1, 0.2, 0.3, 0.4, 0.5]

# Optimizer
batch_size_list = [10, 50, 100, 500, 1000]  # [100]
lr_list = [[1e-3, 1e-4]]
nepochs_list = [[5000, 40000]]

# Iterate over all these lists and create a list of settings
new_settings_list = []
settings = {"dataset": {}, "architecture": {}, "optimizer": {}}

# Iterate over dataset
for dp, fp in zip(dataset_path, fingerprints_path):
    settings["dataset"].update({"dataset_path": dp, "fingerprints_path": fp})

    # Iterate over architecture
    for Nlayers, Nnodes in zip(Nlayers_list, Nnodes_list):
        for ratio in dropout_list:
            settings["architecture"].update(
                {"Nlayers": Nlayers, "Nnodes": list(Nnodes), "dropout_ratio": ratio}
            )

            # Iterate over optimizer settings
            for batch_size in batch_size_list:
                for lr in lr_list:
                    for nepochs in nepochs_list:
                        settings["optimizer"].update(
                            {
                                "batch_size": batch_size,
                                "learning_rate": lr,
                                "nepochs": nepochs,
                            }
                        )
                        new_settings_list.append(copy.deepcopy(settings))

# Export the settings
# Import default settings
with open(SETTINGS_DIR / "settings0.json", "r") as f:
    settings0 = json.load(f)
# Import the previously exported settings list. This file is used to prevent duplicates
old_settings_file = SETTINGS_DIR / "settings_list.json"
if old_settings_file.exists():
    with open(old_settings_file, "r") as f:
        old_settings_dict = json.load(f)
else:
    old_settings_dict = {"settings0": settings0}  # Initialize with the default_settings
istart = len(old_settings_dict)  # Index to start naming the keys of settings dictionary
# Iterate to remove duplicates from the new settings list
for val in old_settings_dict.values():
    try:
        idx = new_settings_list.index(val)
        new_settings_list.pop(idx)
    except ValueError:
        # The new list doesn't contain the old settings
        pass
# Update the settings dictionary
new_settings_dict = copy.deepcopy(old_settings_dict)
for ii, settings in enumerate(new_settings_list):
    new_settings_dict.update({f"settings{istart+ii}": settings})
# Export the new settings dictionary
with open(old_settings_file, "w") as f:
    json.dump(new_settings_dict, f, indent=4)


# Iterate over the settings and submit jobs
SETTINGS_DIR = WORK_DIR / "settings"  # Folder to store all the settings json files
if not SETTINGS_DIR.exists():
    SETTINGS_DIR.mkdir()

for name, settings in new_settings_dict.items():
    # Write settings file
    settings_path = SETTINGS_DIR / f"{name}.json"
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)

    # Let's also prepare the settings JSON files using the test set, which will be used
    # to evaluate the loss and find the best model
    settings_test = copy.deepcopy(settings)
    settings_test["dataset"]["dataset_path"] = (
        "/home/yonatank/nobackup/autodelete/compare_UQ/neuralnetwork/C/data/mingjian_partition_data/carbon_test_set",
    )[0]

    with open(SETTINGS_DIR / f"{name}_test.json", "w") as f:
        json.dump(settings_test, f, indent=4)

    # Command line argument
    args_str = f"--settings-path {settings_path}"

    # Generate a slurm command
    slurm_commands = """#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=150:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores
#SBATCH --nodes=1   # number of node
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH --job-name=DUNN_train_%s   # job name
#SBATCH --mail-user=kurniawanyo@outlook.com   # email address
#SBATCH --mail-type=FAIL
#SBATCH -o ./results/%s/training.out # STDOUT


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source ~/.bash_profile
# echo "Load modules"
# module purge
# module restore kim_project_24

# source /fslhome/yonatank/local/bin/kim-api-activate
# source /fslhome/yonatank/myenv3.9/bin/activate

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn.
# Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUM_CORES=${SLURM_NTASKS}
echo "Running ${SLURM_JOB_NAME}"
echo "Running on ${NUM_CORES} cores"

time python training.py %s

echo "All Done!"
    """ % (
        name,
        name,
        args_str,
    )

    # Write
    slurm_file = "training_submit_job.sh"
    # with open(slurm_file, "w") as f:
    #     f.write(slurm_commands)
    # Submit job
    # subprocess.run(["sbatch", slurm_file])
