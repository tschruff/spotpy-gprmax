#!/bin/bash -x

# job name
#SBATCH --job-name=spotpy-gprmax-gpu

# standard and error output files
# %j will be replaced with the unique job id
#SBATCH --output=gpu_%j_out.txt
#SBATCH --error=gpu_%j_err.txt

# maximum wallclock time hours:minutes:seconds
# When the runtime exceeds this, the job is killed with no mercy
#SBATCH --time=01:59:00

# The number of compute nodes to use.
# JURECA currently has 1872 nodes
# of which ONLY 75 nodes are equipped with NIVIDIA GPUs.
# When you want to use GPUs, "--nodes" must not be larger than 75.
#SBATCH --nodes=1

# Number of MPI tasks per node.
# NOTE: ntasks-per-node is optional
#SBATCH --ntasks-per-node=4

# Number of total MPI tasks: ntasks = nodes * ntasks-per-node
# NOTE: ntasks must be at least (NUM_SPOTPY_WORKERS * 2) + 1 (see run.py for details)
#SBATCH --ntasks=4

# Number of cores per MPI task (OpenMP); minimise when using GPU mode
# NOTE: JURECA has 2 cpus per node with 12 physical and 24 logical (hyperthreading) cores per cpu
#       That means you could use up to 48 logical cores on one node.
# NOTE: Also notice that "--cpus-per-task" is misleading as it actually specifies "--cores-per-tasks"
#SBATCH --cpus-per-task=2

# Number of GPUs to use
# Comment out (witch at least two ##) if you don't want to use GPU mode
#SBATCH --gres=gpu:4

# Partition/queue, must be set to "gpus" when using GPUs.
# For info on available partitions go to
# https://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JURECA/UserInfo/QuickIntroduction.html?nn=1803700
#SBATCH --partition=gpus

# Compute project
#SBATCH --account=slts

# load modules
source env2019a.sh

# activate virtual Python environment
source venv/bin/activate

# Set number of OpenMP threads (can minimise if solving with GPU)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

NUM_SPOTPY_WORKERS=5

# add --mpi=pmi2 after srun to enable mpi2 support
srun -n ${SLURM_NTASKS} python run.py --model "cylinder_Bscan_2D" --ntraces 10 --nrep 20 --ncplx 5 --logfile --verbose ${NUM_SPOTPY_WORKERS}
