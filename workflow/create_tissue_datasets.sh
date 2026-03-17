#!/bin/bash
set -euo pipefail

# Run this script inside the opensp4 environment (e.g., `conda activate opensp4`).

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$WORKFLOW_DIR")"

TISSUE_GFF_DIR="${TISSUE_GFF_DIR:-$PROJECT_DIR/new_data/tissue_gtf}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$PROJECT_DIR/new_data/tissue_dataset}"
GENOME_FASTA="${GENOME_FASTA:-/home1/xyf/data/openspliceai_data/fasta/genome.fa}"

PARSE_TYPE="${PARSE_TYPE:-canonical}"
BIOTYPE="${BIOTYPE:-protein-coding}"
CHR_SPLIT="${CHR_SPLIT:-train-test}"
SPLIT_METHOD="${SPLIT_METHOD:-human}"
SPLIT_RATIO="${SPLIT_RATIO:-0.8}"
VAL_SPLIT_RATIO="${VAL_SPLIT_RATIO:-0.1}"
FLANKING_SIZE="${FLANKING_SIZE:-10000}"
MIN_IDENTITY="${MIN_IDENTITY:-0.8}"
MIN_COVERAGE="${MIN_COVERAGE:-0.5}"

if ! command -v ecasp >/dev/null 2>&1; then
  echo "ECASP command \`ecasp\` not found in PATH. Activate the opensp4 environment first." >&2
  exit 1
fi

OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR%/}"

declare -a GFF_FILES=()
if [[ "$#" -gt 0 ]]; then
  for tissue in "$@"; do
    gff="${TISSUE_GFF_DIR}/${tissue}/${tissue}_step3.gff3"
    if [[ ! -s "$gff" ]]; then
      echo "Missing GFF3 for tissue: $tissue -> $gff" >&2
      exit 1
    fi
    GFF_FILES+=("$gff")
  done
else
  while IFS= read -r -d '' file; do
    GFF_FILES+=("$file")
  done < <(find "$TISSUE_GFF_DIR" -mindepth 2 -maxdepth 2 -type f -name '*_step3.gff3' -print0 | sort -z)
fi

if [[ "${#GFF_FILES[@]}" -eq 0 ]]; then
  echo "No *_step3.gff3 files found under: $TISSUE_GFF_DIR" >&2
  exit 1
fi

echo "=========================================="
echo "ECASP create-data for tissue datasets"
echo "Tissue GFF dir : $TISSUE_GFF_DIR"
echo "Output base    : $OUTPUT_BASE_DIR"
echo "Genome FASTA   : $GENOME_FASTA"
echo "=========================================="

for gff in "${GFF_FILES[@]}"; do
  tissue_dir="$(dirname "$gff")"
  tissue="$(basename "$tissue_dir")"
  output_dir="${OUTPUT_BASE_DIR}/${tissue}/"

  echo ""
  echo "[Tissue] $tissue"
  echo "  GFF3   : $gff"
  echo "  Output : $output_dir"

  mkdir -p "$output_dir"

  ecasp create-data \
    --annotation-gff "$gff" \
    --genome-fasta "$GENOME_FASTA" \
    --output-dir "$output_dir" \
    --parse-type "$PARSE_TYPE" \
    --biotype "$BIOTYPE" \
    --chr-split "$CHR_SPLIT" \
    --split-method "$SPLIT_METHOD" \
    --split-ratio "$SPLIT_RATIO" \
    --val_split_ratio "$VAL_SPLIT_RATIO" \
    --flanking-size "$FLANKING_SIZE" \
    --verify-h5 \
    --remove-paralogs \
    --min-identity "$MIN_IDENTITY" \
    --min-coverage "$MIN_COVERAGE"
done

echo ""
echo "✅ All tissue datasets are generated under: $OUTPUT_BASE_DIR"
