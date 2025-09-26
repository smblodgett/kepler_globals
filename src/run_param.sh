#!/bin/bash --login

#SBATCH --time=12:00:00
#SBATCH --ntasks=501
#SBATCH --mem-per-cpu=4G
#SBATCH -J "kepler_globals_param"

ulimit -n 65535

mamba activate kepler_globals

echo "warming python imports on each compute node..."
srun --exclusive -u python - <<'PY'
import time, socket
t0 = time.time()
# third-party libs
import numpy, scipy, pandas, numba, matplotlib, seaborn, emcee, mpi4py, schwimmbad, json
# local kg modules (import the exact names you use in kg_run_param.py)
import kg_likelihood
import kg_griddefiner
import kg_param_boundary_arrays
import kg_param_initial_guess
import kg_utilities
import kg_probability_distributions
import kg_plots
print("warm imports done on", socket.gethostname(), "took", time.time()-t0)
PY

python kg_initialize_voxel_grid.py

echo "beginning mpiexec"

mpirun -np $SLURM_NTASKS python -u kg_run_param.py 0

echo "something here to combine the results"