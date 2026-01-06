#!/bin/bash
# Batch evaluation script for celeba64_uvit_small_cs curriculum learning checkpoints
# Evaluates all available checkpoints in order

set -e  # Exit on error

# Get current timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Configuration
CONFIG_FILE="configs/celeba64_uvit_small.py"
NUM_GPUS=6
CHECKPOINT_DIR="/home/sunj11/Documents/U-ViT-fresh/workdir/celeba64_uvit_small_cs/default_20260101_073037/ckpts"
RESULTS_DIR="eval_results/celeba64_uvit_small_cs/${TIMESTAMP}"
SAMPLES_DIR="eval_samples/celeba64_uvit_small_cs/${TIMESTAMP}"

# Auto-detect all checkpoints
# CHECKPOINTS=(150000 160000 170000 180000 190000)
CHECKPOINTS=(10000  30000  50000  70000 90000 110000 130000 150000 170000 190000)


# Create output directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$SAMPLES_DIR"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/fid_summary.txt"
echo "FID Evaluation Summary - $(date)" > "$SUMMARY_FILE"
echo "======================================" >> "$SUMMARY_FILE"
echo "Config: $CONFIG_FILE" >> "$SUMMARY_FILE"
echo "Checkpoint Dir: $CHECKPOINT_DIR" >> "$SUMMARY_FILE"
echo "======================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Function to extract FID score from log file
extract_fid() {
    local log_file=$1
    grep "fid=" "$log_file" | tail -1 | sed 's/.*fid=\([0-9.]*\).*/\1/'
}

# Record start time
START_TIME=$(date +%s)

# Evaluate each checkpoint
for ckpt in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating checkpoint: ${ckpt}"
    echo "=========================================="

    # Define paths
    CKPT_PATH="${CHECKPOINT_DIR}/${ckpt}.ckpt"

    # Check for EMA model first, fallback to regular model
    if [ -f "${CKPT_PATH}/nnet_ema.pth" ]; then
        NNET_PATH="${CKPT_PATH}/nnet_ema.pth"
        MODEL_TYPE="ema"
        echo "Using EMA model"
    elif [ -f "${CKPT_PATH}/nnet.pth" ]; then
        NNET_PATH="${CKPT_PATH}/nnet.pth"
        MODEL_TYPE="nnet"
        echo "Using standard model"
    else
        echo "ERROR: Checkpoint not found: $CKPT_PATH"
        echo "Checkpoint ${ckpt}: NOT FOUND" >> "$SUMMARY_FILE"
        continue
    fi

    SAMPLE_PATH="${SAMPLES_DIR}/${ckpt}_${MODEL_TYPE}/"
    LOG_FILE="${RESULTS_DIR}/eval_${ckpt}_${MODEL_TYPE}.log"

    # Create sample directory
    mkdir -p "$SAMPLE_PATH"

    # Run evaluation
    echo "Running: accelerate launch --multi_gpu --num_processes $NUM_GPUS --mixed_precision fp16 eval.py ..."
    echo "  Config: $CONFIG_FILE"
    echo "  Checkpoint: $NNET_PATH"
    echo "  Samples will be saved to: $SAMPLE_PATH"
    echo "  Log file: $LOG_FILE"
    echo ""

    # Activate conda and run
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate UVIT && \
    accelerate launch --multi_gpu --num_processes $NUM_GPUS --mixed_precision fp16 \
        eval.py \
        --config="$CONFIG_FILE" \
        --nnet_path="$NNET_PATH" \
        --config.sample.path="$SAMPLE_PATH" \
        --output_path="$LOG_FILE" || {
            echo "ERROR: Evaluation failed for checkpoint ${ckpt}"
            echo "Checkpoint ${ckpt} (${MODEL_TYPE}): FAILED" >> "$SUMMARY_FILE"
            continue
        }

    # Extract and display FID score
    FID_SCORE=$(extract_fid "$LOG_FILE")
    echo ""
    echo "Checkpoint ${ckpt} (${MODEL_TYPE}) - FID Score: $FID_SCORE"
    echo "Checkpoint ${ckpt} (${MODEL_TYPE}): FID = $FID_SCORE" >> "$SUMMARY_FILE"
    echo ""
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="
echo ""
echo "Summary:"
cat "$SUMMARY_FILE"
echo ""
echo "Generated images saved to: $SAMPLES_DIR/"
echo "Log files saved to: $RESULTS_DIR/"
echo "Summary saved to: $SUMMARY_FILE"
