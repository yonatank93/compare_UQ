#!/bin/bash

#SBATCH --time=70:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=500G   # memory per CPU core
#SBATCH -J "FIM_196_196_128"   # job name
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
export NUM_CORES=${SLURM_NTASKS} # Default empty
export MKL_NUM_THREADS=$SLURM_CPUS_ON_NODE # Default 1
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_ON_NODE # Default empty
# export OPENBLAS_NUM_THREADS=$SLURM_CPUS_ON_NODE # Default 1
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_ON_NODE # Default empty
echo "Running ${SLURM_JOB_NAME}"
echo "Running on ${NUM_CORES} cores"

time python fim_entire_network.py --partition mingjian --nlayers 4 --nnodes 196 196 128
