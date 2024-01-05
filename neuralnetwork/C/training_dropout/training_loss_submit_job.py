"""Submit job to train the NN model."""

from pathlib import Path
import sys
import json
import jinja2
import subprocess


# Command line argument, specifying to use training or test set
argv = sys.argv
config_id = argv[1]  # "train" or "test"

# Read setting file
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
with open(ROOT_DIR / "settings.json", "r") as f:
    settings = json.load(f)
partition = settings["partition"]


slurm_tpl = """#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=165:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH --job-name=DUNN_loss_epochs_{{ config_id }}   # job name
#SBATCH --mail-user=kurniawanyo@outlook.com   # email address
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
echo "Load modules"
module purge
module restore kim_project

source /fslhome/yonatank/local/bin/kim-api-activate
source /fslhome/yonatank/myenv3.8/bin/activate

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUM_CORES=${SLURM_NTASKS}
echo "Running ${SLURM_JOB_NAME}"
echo "Running on ${NUM_CORES} cores"

python training_loss_evaluate_loss.py {{ config_id }}

echo "All Done!"
"""

# Render
env = jinja2.Environment()
template = env.from_string(slurm_tpl)
slurm_file = "training_submit_job.sh"
slurm_commands = template.render(config_id=config_id)
# Write
with open(slurm_file, "w") as f:
    f.write(slurm_commands)
# Submit job
subprocess.run(["sbatch", slurm_file])
