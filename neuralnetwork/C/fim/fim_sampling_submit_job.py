import jinja2
import subprocess
import re
import json

tpl = """#!/bin/bash

#SBATCH --time=50:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=50G   # memory per CPU core
#SBATCH --job-name={{ jobname }}   # job name
#SBATCH --mail-user=yonatank@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES
source ~/.bash_profile
# echo "Load modules"
# module purge
# module restore kim_project_24

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn.
# Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUM_CORES=${SLURM_NTASKS}
export MKL_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_ON_NODE
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_ON_NODE
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_ON_NODE
echo "Running ${SLURM_JOB_NAME}"
echo "Running on ${NUM_CORES} cores"

time python {{ filename }} {{ args }}
"""


# Iterables
nnodes_try = [64, 128, 196]
nnodes_comb = [
    [n1, n2, n3] for n1 in nnodes_try for n2 in nnodes_try for n3 in nnodes_try
]

args_list = [
    {"partition": "mingjian", "nlayers": 4, "nnodes": " ".join([str(n) for n in nnodes])}
    for nnodes in nnodes_comb
]

filename_list = [
    "fim_effective_parameters_no_prior.py",
    "fim_effective_parameters_weak_prior.py",
    "fim_effective_parameters_strong_prior.py",
    # "fim_all_parameters_no_prior.py",
    # "fim_all_parameters_weak_prior.py",
    # "fim_all_parameters_strong_prior.py",
]

# Run
# Use Jinja templating
env = jinja2.Environment()
template = env.from_string(tpl)

jobs_info = {}
for filename in filename_list:
    for args in args_list:
        jobname = "FIM_sampling_" + args["nnodes"].replace(" ", "_") + filename[3:-3]
        # print(jobname)
        args_str = " ".join([f"--{key} {val}" for key, val in args.items()])
        # print(args_str)

        # Write job script
        content = template.render(jobname=jobname, filename=filename, args=args_str)
        # print(content)
        slurm_file = "fim_sampling_submit_job.sh"
        with open(slurm_file, "w") as f:
            f.write(content)
        # Submit
        process = subprocess.run(
            ["sbatch", slurm_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # Retrieve the job id
        stdout = process.stdout.decode("utf-8")
        print(stdout)
        jobid = int(re.findall("\d+", stdout)[0])
        # Store job info in a dictionary
        jobs_info.update({jobid: {"script": filename, "arguments": args_str}})

# Write JSON file containing jobs info
with open("jobs/fim_sampling_jobs_info.json", "w") as f:
    json.dump(jobs_info, f, indent=4)
