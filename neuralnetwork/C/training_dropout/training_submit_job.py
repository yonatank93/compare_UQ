"""Submit job to train the NN model."""

from pathlib import Path
import json
import subprocess


# Read setting file
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
with open(ROOT_DIR / "settings.json", "r") as f:
    settings = json.load(f)
partition = settings["partition"]
Nnodes = settings["Nnodes"]


slurm_commands = """#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=165:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH --job-name=DUNN_train   # job name
#SBATCH --mail-user=kurniawanyo@outlook.com   # email address
#SBATCH --mail-type=FAIL
#SBATCH -o ./results/training/%s_partition_%s/train_loss.out # STDOUT


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

python training.py

echo "All Done!"
""" % (
    partition,
    "_".join([str(n) for n in Nnodes]),
)

# Write
slurm_file = "training_submit_job.sh"
with open(slurm_file, "w") as f:
    f.write(slurm_commands)
# Submit job
subprocess.run(["sbatch", slurm_file])
