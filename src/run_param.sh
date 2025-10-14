#!/bin/bash --login

#SBATCH --time=8:00:00
#SBATCH --ntasks=2001
#SBATCH --mem-per-cpu=2G
#SBATCH -J "kepler_globals_param"

ulimit -n 65535

mamba activate kepler_globals

echo "warming python imports on each compute node..."
srun --exclusive -u python - <<'PY'
import time, socket, random
t0 = time.time()
# third-party libs
time.sleep(random.uniform(0,15))
import numpy, scipy, pandas, numba, matplotlib, seaborn, emcee, mpi4py, schwimmbad, json
# local kg modules
import kg_likelihood
import kg_griddefiner
import kg_param_boundary_arrays
import kg_param_initial_guess
import kg_utilities
import kg_probability_distributions
import kg_plots
print("warm imports done on", socket.gethostname(), "took", time.time()-t0)
PY

python -u kg_initialize_voxel_grid.py

echo "beginning srun"

# mpirun -np $SLURM_NTASKS python kg_run_param.py 0
srun -n $SLURM_NTASKS --mpi=pmix python -u kg_run_param.py 0

echo "finished MCMC. Beginning plotting!"

python kg_plots.py param_analysis

rm model_run_folder.json