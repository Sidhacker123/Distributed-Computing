#!/bin/bash

# Optional: activate your virtualenv if needed
# source ~/envs/gpu-env/bin/activate

echo "Starting Distributed Matrix Compute Run..."

# Install dependencies if not installed
echo "Installing Python dependencies..."
pip install -r requirements.txt > /dev/null

# Set matrix size and GPU count
MATRIX_SIZE=4096
NUM_GPUS=4
PARTITION="row"
PROFILE_FLAG="--profile"

# Generate timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="run_output_${TIMESTAMP}.log"

# Run main.py
echo "Running matrix multiplication with ${NUM_GPUS} GPUs..."
python3 project/main.py \
    --m $MATRIX_SIZE \
    --n $MATRIX_SIZE \
    --k $MATRIX_SIZE \
    --gpus $NUM_GPUS \
    --partition $PARTITION \
    $PROFILE_FLAG | tee "scripts/$LOGFILE"

echo "Run complete. Output saved to scripts/$LOGFILE"

