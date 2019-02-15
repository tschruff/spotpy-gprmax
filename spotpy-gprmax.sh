#!/bin/bash

# Job name
#SBATCH --job-name=spotpy-gprMax

# Maximum wallclock time hours:minutes:seconds
#SBATCH --time=01:59:00

# Number of nodes to use
#SBATCH --nodes=1

# Number of CPUs per task (OpenMP); minimise when using GPU solving
#SBATCH --cpus-per-task=2

# Number of GPUs to use
###SBATCH --gres=gpu:4

# Partition/queue
#SBATCH --partition=devel

# Load required modules
module load intel-para/2017b.1-mt
module load CUDA/9.0.176

# Set number of OpenMP threads (can minimise if solving with GPU)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export PATH="/homec/paj1805/paj18052/miniconda3/bin:$PATH"
source activate spotpy-gprmax

# Number of MPI tasks for SPOTPY with SCE should be number of complexes (workers) + 1 (master)
srun -N <nodes> hostname
mpiexec.hydra -n 3 ./spotpy_run_gprMax.py
