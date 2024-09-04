"""Submit job to train the NN model."""

import subprocess


# Iterables
# Number of nodes in each hidden layer to try
nnodes_try = [128]
# Combinations of the number of nodes
nnodes_comb = [
    [n1, n2, n3] for n1 in nnodes_try for n2 in nnodes_try for n3 in nnodes_try
]
# List of dropout ratio
dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5]

settings_list = [
    {"partition": "mingjian", "Nlayers": 4, "Nnodes": nnodes, "dropout_ratio": ratio}
    for nnodes in nnodes_comb
    for ratio in dropout_list
]

for settings in settings_list[1:]:
    # Read setting file
    partition = settings["partition"]
    Nlayers = settings["Nlayers"]
    Nnodes = settings["Nnodes"]
    dropout_ratio = settings["dropout_ratio"]
    Nnodes_str = "_".join([str(n) for n in Nnodes])
    args_str = (
        f"--partition {partition} "
        + f"--nlayers {Nlayers} "
        + f"--nnodes {' '.join([str(n) for n in Nnodes])} "
        + f"--dropout {dropout_ratio}"
    )

    slurm_commands = """#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=120:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores
#SBATCH --nodes=1   # number of node
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH --job-name=DUNN_train_d%s_%s   # job name
#SBATCH --mail-user=kurniawanyo@outlook.com   # email address
#SBATCH --mail-type=FAIL
#SBATCH -o ./results/training_d%s/%s_partition_%s/train_loss.out # STDOUT


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
        dropout_ratio,
        Nnodes_str,
        dropout_ratio,
        partition,
        Nnodes_str,
        args_str,
    )

    # Write
    slurm_file = "training_submit_job.sh"
    with open(slurm_file, "w") as f:
        f.write(slurm_commands)
    # Submit job
    subprocess.run(["sbatch", slurm_file])
