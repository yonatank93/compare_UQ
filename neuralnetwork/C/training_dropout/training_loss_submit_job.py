"""Submit job to train the NN model."""

from pathlib import Path
import jinja2
import subprocess

# Iterables
# Number of nodes in each hidden layer to try
nnodes_try = [64, 128, 196]
# Combinations of the number of nodes
nnodes_comb = [
    [n1, n2, n3] for n1 in nnodes_try for n2 in nnodes_try for n3 in nnodes_try
]
settings_list = [
    {"partition": "mingjian", "Nlayers": 4, "Nnodes": nnodes} for nnodes in nnodes_comb
]
configs_list = ["train", "test"]


# Slurm job template
slurm_tpl = """#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH --job-name=DUNN_loss_{{ suffix }}_{{ cid }}   # job name
#SBATCH --mail-user=kurniawanyo@outlook.com   # email address
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source ~/.bash_profile
# echo "Load modules"
# module purge
# module restore kim_project_24

# source /fslhome/yonatank/local/bin/kim-api-activate
# source /fslhome/yonatank/myenv3.8/bin/activate

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUM_CORES=${SLURM_NTASKS}
echo "Running ${SLURM_JOB_NAME}"
echo "Running on ${NUM_CORES} cores"

time python training_loss_evaluate_loss.py {{ args_str }}

echo "All Done!"
"""

env = jinja2.Environment()
template = env.from_string(slurm_tpl)
slurm_file = "training_submit_job.sh"

WORK_DIR = Path(__file__).absolute().parent
RES_DIR = WORK_DIR / "results" / "training"

# Submit jobs
for cid in configs_list:
    for settings in settings_list[1:]:
        # Read setting file
        partition = settings["partition"]
        Nnodes = settings["Nnodes"]
        suffix = "_".join([str(n) for n in Nnodes])
        args_str = (
            f"--cid {cid} --partition {settings['partition']} "
            + f"--nlayers {settings['Nlayers']} "
            + f"--nnodes {' '.join([str(n) for n in settings['Nnodes']])}"
        )

        # Check if Job has been submitted
        folder = RES_DIR / f"{partition}_partition_{suffix}"
        filepath = folder / f"loss_values_{cid}.txt"
        if not filepath.exists():
            print(suffix, cid)

            # Render
            slurm_commands = template.render(suffix=suffix, cid=cid, args_str=args_str)
            # Write
            with open(slurm_file, "w") as f:
                f.write(slurm_commands)
            # Submit job
            subprocess.run(["sbatch", slurm_file])
