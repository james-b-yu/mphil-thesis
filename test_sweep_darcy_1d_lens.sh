#!/bin/bash

# Define the list of len methods to loop through
lens=("0005" "001" "002" "005" "010" "020" "050")

# Outer loop for len methods
for len in "${lens[@]}"
do
  # Inner loop for the run index (0, 1, 2)
  for i in {0..2}
  do
     echo "==> Running test with len: ${len}, run: ${i}"
     ./run.py test \
         --config=./configurations/sweep_darcy_1d/${len}.yml \
         --pth=./out_darcy_1d_${len}/ema_epoch_1500.pth \
         --n-steps=100 \
         --n-batch-size=1500 \
         --out-file=./test_sweep_darcy_1d/final_sweep-${len}-${i}_epoch_1500.npz \
         --stats-out=./test_sweep_darcy_1d/final_sweep-${len}-${i}_epoch_1500.csv \
         --weighting=linear-out \
         --method=em_2
  done
done