#!/bin/bash
for i in $(seq 0 3419); do
    python kg_run.py "$i"
done