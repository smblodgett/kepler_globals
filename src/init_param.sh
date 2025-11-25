#!/bin/bash --login

#SBATCH --time=8:00:00
#SBATCH --ntasks=1001
#SBATCH --mem-per-cpu=4G
#SBATCH -J "kepler_globals_init"

export PMIX_MCA_psec=^munge

ulimit -n 65535

mamba activate kepler_globals

srun -n $SLURM_NTASKS --mpi=pmix python -u kg_initialize_voxel_grid.py