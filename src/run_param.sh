#!/bin/bash --login

#SBATCH --time=12:00:00
#SBATCH --ntasks=2001
#SBATCH --mem-per-cpu=4G
#SBATCH -J "kepler_globals_param"

mamba activate kepler_globals

#rm ../runs/param_model_log.txt


echo "preloading libraries"

python -c "import numpy, scipy, pandas, emcee, json, schwimmbad, mpi4py, corner, matplotlib, seaborn, numba"

python kg_initialize_voxel_grid.py

echo "beginning mpiexec"

mpiexec -n $SLURM_NTASKS python kg_run_param.py 0

count=$(find . -type f -name "param_model_0.h5" | wc -l)

cp ../results/param_backend/param_model_0.h5 ../results/param_backend/param_model_0_$(date +%Y%m%d)_$count.h5