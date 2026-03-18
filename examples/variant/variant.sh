#!/bin/bash
# A script to run the ECASP variant command via `ecasp variant`

# Resolve the parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# # Ensure the ECASP package (`ecasp`) is installed
# pip install $PDIR

# Define required arguments
REF_GENOME_PATH="$PDIR/data/ref_genome/homo_sapiens/GRCh37/hg19.fa"
MODEL_PATH="$PDIR/models/spliceai-mane/400nt/model_400nt_rs14.pt"
INPUT_PATH="$PDIR/data/vcf/input.vcf"
OUTPUT_PATH="$PDIR/examples/variant/output.vcf"
FLANKING_SIZE=400
MODEL_TYPE="pytorch"
ANNOTATION_PATH="$PDIR/examples/data/grch37.txt" # NOTE: this is a custom annotation file

# Build the variant command
CMD="ecasp variant -R "$REF_GENOME_PATH" -A "$ANNOTATION_PATH" -m "$MODEL_PATH" -f $FLANKING_SIZE -t "$MODEL_TYPE" -I "$INPUT_PATH" -O "$OUTPUT_PATH""

# Run the command
echo $CMD 
$CMD 
