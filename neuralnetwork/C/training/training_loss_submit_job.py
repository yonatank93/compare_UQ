"""Submit job to evaluate loss of the NN model."""

from pathlib import Path
from glob import glob
import jinja2
import re
import subprocess


WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
SETTINGS_DIR = ROOT_DIR / "settings"

# Iterables - Get the files listed in the settings folder
settings_files = [str(SETTINGS_DIR / f"settings{ii}.json") for ii in range(13)]

# Slurm job template
slurm_tpl = """#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH --job-name=DUNN_loss_{{ suffix }}   # job name
#SBATCH --mail-user=kurniawanyo@outlook.com   # email address
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source ~/.bash_profile
# echo "Load modules"
# module purge
# module restore kim_project_24

# source /fslhome/yonatank/local/bin/kim-api-activate
# source /fslhome/yonatank/myenv3.8/bin/activate

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn.
# Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUM_CORES=${SLURM_NTASKS}
echo "Running ${SLURM_JOB_NAME}"
echo "Running on ${NUM_CORES} cores"

time python training_loss.py {{ args_str }}

echo "All Done!"
"""

env = jinja2.Environment()
template = env.from_string(slurm_tpl)
slurm_file = "training_loss_submit_job.sh"


# Submit jobs
for settings_path in settings_files:
    # Use regular expression to find settings name
    match = re.search(r"settings\d+", settings_path)
    name = match.group(0)
    RES_DIR = WORK_DIR / "results" / name  # Directory to store the results

    # Prepare the information to input into the job script template
    args_str = f"--settings-path {settings_path}"
    suffix = Path(settings_path).with_suffix("").name

    # Check if Job has been submitted
    cid = "test" if "test" in suffix else "train"
    filepath = RES_DIR / f"loss_values_{cid}.txt"
    if not filepath.exists():
        print(suffix, cid)

        # Render
        slurm_commands = template.render(suffix=suffix, args_str=args_str)
        # Write
        with open(slurm_file, "w") as f:
            f.write(slurm_commands)
        # Submit job
        subprocess.run(["sbatch", slurm_file])
