#!/bin/bash --login

#SBATCH --time=1:00:00
#SBATCH --ntasks=501
#SBATCH --mem-per-cpu=4G
#SBATCH -J "kepler_globals_param_mpitesting"

ulimit -n 65536       # increase open files
ulimit -u 131072      # increase max user processes (if allowed)
ulimit -c unlimited   # keep core dumps

mamba activate kepler_globals

mpirun --report-bindings --mca mpi_preconnect_mpi 0 -np $SLURM_NTASKS python -c "import mpi4py; import pprint; pprint.pprint(mpi4py.get_config()); print('mpi4py __file__ ->', mpi4py.__file__)"

echo "yay, we're done!"