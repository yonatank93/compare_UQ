from pathlib import Path
import jinja2
import subprocess

WORK_DIR = Path(__file__).absolute().parent
RES_DIR = WORK_DIR / "results" / "bootstrap"


slurm_tpl = """#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=70:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH --gpus=1  # Use 1 GPU
#SBATCH --job-name=DUNN_bootstrap_{{ idx_str }}   # job name
#SBATCH --mail-user=kurniawanyo@outlook.com   # email address
#SBATCH --mail-type=FAIL
#SBATCH -o ./results/bootstrap/{{ idx_str }}/train_loss_3.out # STDOUT


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

python bootstrap_single_set.py {{ set_idx }}

echo "All Done!"
"""

env = jinja2.Environment()
template = env.from_string(slurm_tpl)
fname = "submit_job_bootstrap.sh"

nsamples = 100
for ii in range(nsamples):
    # Make directoryto store the result for sample ii
    SAMPLE_DIR = RES_DIR / f"{ii:03d}"
    if not SAMPLE_DIR.exists():
        SAMPLE_DIR.mkdir(parents=True)
    # Render
    content = template.render(set_idx=ii, idx_str=SAMPLE_DIR.name)
    # Write sbatch file
    with open(fname, "w") as f:
        f.write(content)
    # Submit job
    subprocess.run(["sbatch", fname])
