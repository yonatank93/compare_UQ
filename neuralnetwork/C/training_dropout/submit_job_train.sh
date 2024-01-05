#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=165:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH --job-name=DUNN_train   # job name
#SBATCH --mail-user=kurniawanyo@outlook.com   # email address
#SBATCH --mail-type=FAIL
#SBATCH -o ./results/loss/train_loss_1.out # STDOUT


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
echo "Load modules"
module purge
module restore kim_project

source /fslhome/yonatank/local/bin/kim-api-activate
source /fslhome/yonatank/myenv3.8/bin/activate

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUM_CORES=${SLURM_NTASKS}
echo "Running ${SLURM_JOB_NAME}"
echo "Running on ${NUM_CORES} cores"

python initial_training.py

echo "All Done!"
