#!/bin/bash
for i in $(seq 3047 3057); do
    python kg_run.py "$i"
done