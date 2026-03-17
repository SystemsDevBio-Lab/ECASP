#!/bin/bash
# A script to run the ECASP training command via `ecasp train`

# Resolve the parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

RANDOM_SEED=11

SPECIES="MANE"
FLANKING_SIZE="10000"

# Set up output directory
OUTPUT_DIR="$PDIR/examples/train/results/"
mkdir -p $OUTPUT_DIR

# Set up dataset paths (adjust these as needed for each species)
TRAIN_DATASET="$PDIR/examples/create-data/results/dataset_train.h5"
TEST_DATASET="$PDIR/examples/create-data/results/dataset_test.h5"

OUTPUT_FILE="$OUTPUT_DIR/output.log"
ERROR_FILE="$OUTPUT_DIR/error.log"

# Run the ECASP training command
echo ecasp train --flanking-size $FLANKING_SIZE \
    --train-dataset $TRAIN_DATASET \
    --test-dataset $TEST_DATASET \
    --output-dir $OUTPUT_DIR \
    --project-name ${SPECIES}_train \
    --random-seed ${RANDOM_SEED} \
    --scheduler CosineAnnealingWarmRestarts \
    --loss cross_entropy_loss > $OUTPUT_FILE 2> $ERROR_FILE
