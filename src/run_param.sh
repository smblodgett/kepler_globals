#!/bin/bash --login

#SBATCH --time=24:00:00
#SBATCH --ntasks=1000
#SBATCH --mem=20G
#SBATCH -J "kepler_globals_param"
#SBATCH --qos=physics

mamba activate kepler_globals

rm ../runs/param_model_log.txt

mpiexec -n $SLURM_NTASKS python kg_run_param.py 0

count=$(find . -type f -name "myfile.txt" | wc -l)

cp ../results/param_backend/param_model_0.h5 ../results/param_backend/param_model_0_$(date +%Y%m%d)_$count.h5