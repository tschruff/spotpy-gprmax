# load modules
module --force purge
module use $OTHERSTAGES
module load Stages/2019a
module load GCCcore/.8.3.0
module load Python/3.6.8
module load intel-para/2019a
module load CUDA/10.1.105
