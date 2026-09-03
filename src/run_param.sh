#!/bin/bash --login

#SBATCH --time=12:00:00
#SBATCH --ntasks=101
#SBATCH --mem-per-cpu=10G
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH -J "kepler_globals_param"

export PMIX_MCA_psec=^munge

ulimit -n 65535

mamba activate kepler_globals

echo "warming python imports on each compute node..."
srun --exclusive -u python - <<'PY'
import time, socket, random
t0 = time.time()
# third-party libs
time.sleep(random.uniform(0,10))
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

echo "beginning srun"

model_id=1

# mpirun -np $SLURM_NTASKS python kg_run_param.py 0
srun -n $SLURM_NTASKS --mpi=pmix python -u kg_run_param.py $model_id

echo "finished MCMC. Beginning plotting!"

python kg_plots.py param_analysis

rm model_run_folder.json