#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=120:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores
#SBATCH --nodes=1   # number of node
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH --job-name=DUNN_train_d0.5_128_128_128   # job name
#SBATCH --mail-user=kurniawanyo@outlook.com   # email address
#SBATCH --mail-type=FAIL
#SBATCH -o ./results/training_d0.5/mingjian_partition_128_128_128/train_loss.out # STDOUT


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source ~/.bash_profile
# echo "Load modules"
# module purge
# module restore kim_project_24

# source /fslhome/yonatank/local/bin/kim-api-activate
# source /fslhome/yonatank/myenv3.9/bin/activate

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn.
# Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUM_CORES=${SLURM_NTASKS}
echo "Running ${SLURM_JOB_NAME}"
echo "Running on ${NUM_CORES} cores"

time python training.py --partition mingjian --nlayers 4 --nnodes 128 128 128 --dropout 0.5

echo "All Done!"
    