#!/bin/bash
start=$SECONDS
rm ../runs/grid_model_log.txt

for i in $(seq 2000 3419); do
    python kg_run_grid.py "$i" && \
    python kg_plots.py trace "$i" && \
    python kg_plots.py grid_corner "$i" 
done

python kg_plots.py heatmap 0

duration=$((SECONDS-start))
let "hours=duration/3600"
let "minutes=(duration%3600)/60"
let "seconds=(duration%3600)%60"
echo "Grid Model completed in $hours hour(s), $minutes minute(s) and $seconds second(s)" 