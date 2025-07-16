#!/bin/bash --login

#SBATCH --time=24:00:00
#SBATCH --ntasks=200
#SBATCH --nodes=1
#SBATCH -- mem-per-cpu=1024M
#SBATCH -J "kepler_globals_param"
#SBATCH --qos=physics


rm ../runs/param_model_log.txt

srun --mpi=pmix_v3 -n $SLURM_NTASKS python kg_run_param.py 0 

# python kg_run_param.py "$i" # && \python kg_plots.py "$i" trace && \python kg_plots.py "$i" corner

# #python kg_plots.py 0 heatmap

# duration=$((SECONDS-start))
# let "hours=duration/3600"
# let "minutes=(duration%3600)/60"
# let "seconds=(duration%3600)%60"
# echo "Parametric Model completed in $hours hour(s), $minutes minute(s) and $seconds second(s)" 