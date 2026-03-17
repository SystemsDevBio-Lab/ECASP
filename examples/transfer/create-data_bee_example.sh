#!/bin/bash
# A script to run the ECASP create-data command via `ecasp create-data`

# Resolve the parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

GENOME="$PDIR/examples/data/honeybee/HAv3.1_genomic_10_sample.fna"
ANNOTATION="$PDIR/examples/data/honeybee/HAv3.1_genomic_10_sample.gff"
OUTPUT_DIR="$PDIR/examples/transfer-learn/results/"

mkdir -p ${OUTPUT_DIR}

# Run the command
ecasp create-data \
    --remove-paralogs \
    --min-identity 0.8 \
    --min-coverage 0.8 \
    --genome-fasta "$GENOME" \
    --annotation-gff "$ANNOTATION" \
    --output-dir "$OUTPUT_DIR" \
    --parse-type canonical \
    --write-fasta > "$OUTPUT_DIR/output.log" 2> "$OUTPUT_DIR/error.log"
