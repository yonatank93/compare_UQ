#!/bin/bash

#SBATCH --time=50:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=50G   # memory per CPU core
#SBATCH --job-name=FIM_sampling_196_196_196_effective_parameters_strong_prior   # job name
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

time python fim_effective_parameters_strong_prior.py --partition mingjian --nlayers 4 --nnodes 196 196 196