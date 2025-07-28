#!/bin/bash

# Define the list of weighting methods to loop through
weighting_methods=("exponential-in" "exponential-in-out" "linear-in" "linear-out" "linear-in-out")

# Outer loop for weighting methods
for weighting in "${weighting_methods[@]}"
do
  # Inner loop for the run index (0, 1, 2)
  for i in {0..2}
  do
     echo "==> Running test with weighting: ${weighting}, run: ${i}"
     ./run.py test \
         --config=./configurations/sweep_darcy_1d/010.yml \
         --pth=./out_darcy_1d_hybrid_new/ema_epoch_1500.pth \
         --n-steps=100 \
         --n-batch-size=1000 \
         --out-file=./test_sweep_darcy_1d/hybrid-${weighting}-${i}.npz \
         --stats-out=./test_sweep_darcy_1d/hybrid-${weighting}-${i}.csv \
         --weighting=${weighting}
  done
done