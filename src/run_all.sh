#!/bin/bash
start=$SECONDS

for i in $(seq 0 3419); do
    python kg_run.py "$i"
done

duration=$((SECONDS-start))
let "hours=duration/3600"
let "minutes=(duration%3600)/60"
let "seconds=(duration%3600)%60"
echo "Completed in $hours hour(s), $minutes minute(s) and $seconds second(s)" 
