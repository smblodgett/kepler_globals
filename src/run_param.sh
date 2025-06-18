#!/bin/bash
start=$SECONDS
rm ../runs/param_model_log.txt

python kg_run_param.py "$i" # && \python kg_plots.py "$i" trace && \python kg_plots.py "$i" corner

#python kg_plots.py 0 heatmap

duration=$((SECONDS-start))
let "hours=duration/3600"
let "minutes=(duration%3600)/60"
let "seconds=(duration%3600)%60"
echo "Parametric Model completed in $hours hour(s), $minutes minute(s) and $seconds second(s)" 