#!/bin/bash
# A script to run the ECASP calibrate command via `ecasp calibrate`

# Resolve the parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

CURRENT_SPECIES="MANE"
FLANKING_SIZE=10000

# Set up output directory
OUTPUT_DIR="$PDIR/examples/calibrate/results/"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/output.log"
ERROR_FILE="$OUTPUT_DIR/error.log"

# Set up dataset paths
# TRAIN_DATASET="$PDIR/examples/create-data/results/dataset_train.h5"
# TEST_DATASET="$PDIR/examples/create-data/results/dataset_test.h5"

TRAIN_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/dataset_train.h5"
TEST_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/dataset_test.h5"
PRETRAINED_MODEL="$PDIR/models/spliceai-mane/10000nt/model_10000nt_rs10.pt"

CURRENT_SPECIES_lowercase="${CURRENT_SPECIES,,}"

echo "Current species: $CURRENT_SPECIES"
echo "Flanking size: $FLANKING_SIZE"
echo "Random seed: $RANDOM_SEED"
echo "Output directory: $OUTPUT_DIR"
echo "Train dataset: $TRAIN_DATASET"
echo "Test dataset: $TEST_DATASET"
echo "Pretrained model: $PRETRAINED_MODEL"

# Run the ECASP calibrate command
ecasp calibrate --flanking-size $FLANKING_SIZE \
    --train-dataset $TRAIN_DATASET \
    --test-dataset $TEST_DATASET \
    --output-dir $OUTPUT_DIR \
    --project-name human_${CURRENT_SPECIES}_calibrate \
    --pretrained-model $PRETRAINED_MODEL \
    --loss cross_entropy_loss > $OUTPUT_FILE 2> $ERROR_FILE
