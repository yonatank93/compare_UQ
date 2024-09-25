from pathlib import Path
import json
import argparse
import jinja2
import subprocess

# Read setting file
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
SETTINGS_DIR = ROOT_DIR / "settings"

# Read command line argument
arg_parser = argparse.ArgumentParser("Settings of the calculations")
arg_parser.add_argument(
    "-s", "--settings-path", default=SETTINGS_DIR / "settings0.json", dest="settings_path"
)
arg_parser.add_argument("-i", "--set-idx", type=int, dest="set_idx")
args = arg_parser.parse_args()
settings_path = Path(args.settings_path)
set_idx = args.set_idx
RES_DIR = WORK_DIR / "results" / settings_path.with_suffix("").name


slurm_tpl = """#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=70:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH --job-name=DUNN_bootstrap_{{ idx_str }}   # job name
#SBATCH --mail-user=kurniawanyo@outlook.com   # email address
#SBATCH --mail-type=FAIL
#SBATCH -o {{ result_dir }}/{{ idx_str }}/train.out # STDOUT


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source ~/.bash_profile
# echo "Load modules"
# module purge
# module restore kim_project_24

source /fslhome/yonatank/local/bin/kim-api-activate
source /fslhome/yonatank/myenv3.8/bin/activate

# Set the max number of threads to use for programs using OpenMP.
# Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUM_CORES=${SLURM_NTASKS}
echo "Running ${SLURM_JOB_NAME}"
echo "Running on ${NUM_CORES} cores"

python bootstrap_single_set.py -i {{ set_idx }}

echo "All Done!"
"""

env = jinja2.Environment()
template = env.from_string(slurm_tpl)

nsamples = 100
for ii in range(nsamples):
    # Make directory to store the result for sample ii
    SAMPLE_DIR = RES_DIR / f"{ii:03d}"
    fname = SAMPLE_DIR / "bootstrap_submit_job.sh"
    if not SAMPLE_DIR.exists():
        SAMPLE_DIR.mkdir(parents=True)
    # Render
    content = template.render(
        set_idx=ii, idx_str=SAMPLE_DIR.name, result_dir=str(RES_DIR)
    )
    # Write sbatch file
    with open(fname, "w") as f:
        f.write(content)
    # Submit job
    subprocess.run(["sbatch", str(fname)])
