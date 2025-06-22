
# Set the starting and ending ID for the loop
START_ID=1
END_ID=100

echo "Starting batch processing from n-id=$START_ID to n-id=$END_ID..."

# Loop from START_ID to END_ID (inclusive)
for (( id=$START_ID; id<=$END_ID; id++ ))
do
  # Print a message to the console to show progress
  echo "-----------------------------------------"
  echo "Running for n-id = $id"
  echo "-----------------------------------------"

  # Construct and execute the command
  ./run.py --config=./configurations/darcy_1d.yml test_one \
    --pth=./out/epoch_2000.pth \
    --n-id=$id \
    --n-repeats=100 \
    --out-file=./samples-$id.npz \
    --all-t=True

  # Optional: Check if the command was successful
  if [ $? -ne 0 ]; then
    echo "Error running for n-id = $id. Aborting."
    exit 1
  fi
done

echo "Batch processing complete."