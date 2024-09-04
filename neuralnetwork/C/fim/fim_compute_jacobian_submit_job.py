from pathlib import Path
import jinja2
import subprocess


# Iterables
# Number of nodes in each hidden layer to try
nnodes_try = [64, 128, 196]
# Combinations of the number of nodes
# nnodes_comb = [
#     [n1, n2, n3] for n1 in nnodes_try for n2 in nnodes_try for n3 in nnodes_try
# ]
nnodes_comb = [[196, 196, 128]]
settings_list = [
    {"partition": "mingjian", "Nlayers": 4, "Nnodes": nnodes} for nnodes in nnodes_comb
]

tpl = """#!/bin/bash

#SBATCH --time=70:00:00   # walltime
#SBATCH --ntasks=20   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=50G   # memory per CPU core
#SBATCH -J "FIM_{{ suffix }}"   # job name
#SBATCH --mail-user=yonatank@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES
source ~/.bash_profile
# echo "Load modules"
# module purge
# module restore kim_project_24

# Set the max number of threads to use for programs using OpenMP.
# Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUM_CORES=${SLURM_NTASKS}
export MKL_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_ON_NODE
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_ON_NODE
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_ON_NODE
echo "Running ${SLURM_JOB_NAME}"
echo "Running on ${NUM_CORES} cores"

time python fim_entire_network.py {{ args_str }}
"""

# Use Jinja templating
env = jinja2.Environment()
template = env.from_string(tpl)
slurm_file = "fim_jacobian_submit_job.sh"

# Iterate over the configs/combinations and submit the jobs
WORK_DIR = Path(__file__).absolute().parent
RES_DIR = WORK_DIR / "results"

# Submit jobs
for settings in settings_list:
    # Read setting file
    partition = settings["partition"]
    Nnodes = settings["Nnodes"]
    suffix = "_".join([str(n) for n in Nnodes])
    args_str = (
        f"--partition {settings['partition']} --nlayers {settings['Nlayers']} "
        + f"--nnodes {' '.join([str(n) for n in settings['Nnodes']])}"
    )

    # Render
    slurm_commands = template.render(suffix=suffix, args_str=args_str)
    # Write
    with open(slurm_file, "w") as f:
        f.write(slurm_commands)
    # Submit job
    print(suffix)
    subprocess.run(["sbatch", slurm_file])
