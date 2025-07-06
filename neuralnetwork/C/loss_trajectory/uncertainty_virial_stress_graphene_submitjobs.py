"""This script is to submit jobs to run MD simulatins to get the data to compute the
virial stress tensor. There should be 100 jobs submitted by running this script, each job
uses different model sample from dropout ensemble.
"""

from pathlib import Path
import json
import re
import argparse
from tqdm import tqdm
import jinja2
import subprocess
import time

# Read settings
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
SETTINGS_DIR = ROOT_DIR / "settings"

# Command line argument
arg_parser = argparse.ArgumentParser("Settings file path")
arg_parser.add_argument(
    "-p", "--path", default=SETTINGS_DIR / "settings0.json", dest="settings_path"
)
args = arg_parser.parse_args()

settings_path = Path(args.settings_path)
with open(settings_path, "r") as f:
    settings = json.load(f)
RES_DIR = WORK_DIR / "results" / re.match(r"^[^_\.]+", settings_path.name).group()


sbatch_template = """#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=11   # number of processor cores
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=2G   # memory per CPU core
#SBATCH --job-name=virial_losstraj_{{ idx }}   # job name
#SBATCH --mail-user=kurniawanyo@outlook.com   # email address
#SBATCH --mail-type=FAIL
#SBATCH -o {{ outfile }} # STDOUT


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
echo "Load modules"
module purge
module restore kim_project

source /fslhome/yonatank/local/bin/kim-api-activate
source /fslhome/yonatank/myenv3.8/bin/activate

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn.
# Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUM_CORES=${SLURM_NTASKS}
echo "Running ${SLURM_JOB_NAME}"
echo "Running on ${NUM_CORES} cores"

python uncertainty_virial_stress_graphene_runmd.py -p {{ settings }} -i {{ idx }}

echo "All Done!"
"""


job_id_list = []
for idx in tqdm(range(100)):
    outdir = RES_DIR / f"{idx:03d}" / "virial_stress_graphene"
    if not outdir.exists():
        outdir.mkdir(parents=True)
    # Render lammps command
    env = jinja2.Environment()
    template = env.from_string(sbatch_template)
    sbatch_out = "./" + str(outdir) + "/slurm-%j.out"
    sbatch_script = template.render(settings=settings_path, idx=idx, outfile=sbatch_out)
    # Write sbatch script
    outfile = outdir / "submit_job.sh"
    with open(outfile, "w") as f:
        f.write(sbatch_script)
    # Submit job and get the job id
    process = subprocess.run(["sbatch", outfile], capture_output=True, text=True)
    print(process.stdout)
    job_id_list.append(int(re.search(r"\d+", process.stdout).group()))


# # Wait for all jobs to complete I will do it by running squeue and check if job is still
# # running. If job is done, then I will remove it from job_id_list
# while True:
#     run_squeue = subprocess.run(
#         "squeue -u yonatank", capture_output=True, text=True, shell=True
#     )
#     squeue_out = run_squeue.stdout

#     for jid in job_id_list:
#         if str(jid) not in squeue_out:
#             # Jow is done, remove the id from job_id_list
#             print(f"Job with id {jid} is completed.")
#             job_id_list.remove(jid)

#     if len(job_id_list) > 0:
#         # All jobs are done
#         break
#     else:
#         # Wait for 1 minute
#         time.sleep(60)
