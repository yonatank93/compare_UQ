from pathlib import Path
import json
import jinja2
import subprocess

# Read setting file
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
with open(ROOT_DIR / "settings.json", "r") as f:
    settings = json.load(f)
partition = settings["partition"]

# Directories
PART_DIR = ROOT_DIR / f"{partition}_partition_data"
RES_DIR = WORK_DIR / "results" / f"{partition}_partition"


slurm_tpl = """#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=165:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH --gpus=1  # Use 1 GPU
#SBATCH --job-name=DUNN_randinit_{{ idx_str }}   # job name
#SBATCH --mail-user=kurniawanyo@outlook.com   # email address
#SBATCH --mail-type=FAIL
#SBATCH -o ./results/{{ partition_dir }}/{{ idx_str }}/train_loss_2.out # STDOUT


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
echo "Load modules"
module purge
module restore kim_project

source /fslhome/yonatank/local/bin/kim-api-activate
source /fslhome/yonatank/myenv3.8/bin/activate

# Set the max number of threads to use for programs using OpenMP.
# Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUM_CORES=${SLURM_NTASKS}
echo "Running ${SLURM_JOB_NAME}"
echo "Running on ${NUM_CORES} cores"

python randinit_single_set.py {{ set_idx }}

echo "All Done!"
"""

env = jinja2.Environment()
template = env.from_string(slurm_tpl)
fname = RES_DIR / "submit_job_randinit.sh"

nsamples = 100
for ii in range(nsamples):
    # Make directoryto store the result for sample ii
    SAMPLE_DIR = RES_DIR / f"{ii:03d}"
    if not SAMPLE_DIR.exists():
        SAMPLE_DIR.mkdir(parents=True)
    # Render
    content = template.render(
        set_idx=ii, idx_str=SAMPLE_DIR.name, partition_dir=f"{partition}_partition"
    )
    # Write sbatch file
    with open(fname, "w") as f:
        f.write(content)
    # Submit job
    subprocess.run(["sbatch", fname])
