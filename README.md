<p align="center">
  <img src="Logo.png" alt="ECASP logo" width="720">
</p>

ECASP (Expression-Conditioned AI for Splicing Prediction) is a splice prediction framework reconstructed and extended in PyTorch, covering the full workflow from raw genomic input to splice-site prediction and variant annotation. This guide follows the two-stage training strategy used in the current Methods: first jointly train ECASP across multiple developmental systems while keeping the FiLM conditional branch active, then freeze FiLM and fine-tune the sequence backbone on the reference dataset under a zero-vector condition. During inference and variant annotation, the true expression vector of the target system is injected again to obtain system-specific predictions.

---

## Environment and Resources

- **Dependencies**: Python >= 3.10, PyTorch (CUDA >= 11.7 recommended for GPU training), NumPy/Pandas/HDF5/pyfaidx, and related packages. Running `pip install -e .` installs the required Python packages.
- **Hardware**: At least 16 GB of GPU memory is recommended for base-model and FiLM fine-tuning. Variant annotation can run on CPU, but it will be slower.
- **Core data**:
  - Reference genome FASTA (for example `data/genome.fa` plus `.fai`).
  - Tissue- or species-specific GTF/GFF annotation files. The repository already ships the 15 final developmental-system annotations under [`data/tissue_gff3/`](data/tissue_gff3).
  - SpliceAI annotation files (for example `data/grch38.txt`) or a custom annotation.
  - A concatenated and standardized RBP+HVG expression matrix, such as [`data/tissue_expression_features_scaled.csv`](data/tissue_expression_features_scaled.csv).

After installation, you can quickly verify the CLI with:

```bash
pip install -e .
ecasp --help
```

---

## Full Workflow Overview

- **create-data**: Prepare HDF5 datasets for both the reference dataset and multiple developmental systems (train/validation/test).
- **prepare-rbp-expression**: Build condition vectors (RBP + HVG) for each developmental system and prepare the zero vector used for reference fine-tuning.
- **train**: Stage 1, jointly train ECASP across multiple developmental systems with both FiLM and backbone enabled.
- **transfer**: Stage 2, start from the best Stage 1 checkpoint, freeze FiLM, and fine-tune non-FiLM parameters on the reference dataset.
- **variant**: Use the final Stage 2 model plus a target-system condition vector for system-specific VCF annotation.
- **predict**: Use the final Stage 2 model plus a target-system condition vector for system-specific splice-site prediction from FASTA input.

The sections below describe each step in detail.

---

## Generate HDF5 Data (`create-data`)

`create-data` performs two tasks: 1) `create_datafile` slices the annotation into sequence windows; 2) `create_dataset` exports HDF5 shards. A typical command is:

```bash
ecasp create-data \
  --annotation-gff data/tissue_gff3/limb.gff3 \
  --genome-fasta data/genome.fa \
  --output-dir /path/dataset_limb \
  --parse-type canonical \
  --biotype protein-coding \
  --chr-split train-test \
  --split-method human \
  --split-ratio 0.8 \
  --val_split_ratio 0.1 \
  --flanking-size 10000 \
  --verify-h5 \
  --remove-paralogs \
  --min-identity 0.8 \
  --min-coverage 0.5
```

The repository includes the 15 final developmental-system GFF3 annotations used to generate the Stage 1 multi-system training data under [`data/tissue_gff3/`](data/tissue_gff3). The larger multi-system HDF5 training data and the reference dataset are better distributed separately via Zenodo.

---

## Build Condition Vectors (`prepare-rbp-expression`)

Before Stage 1 multi-system joint training, you first need a condition vector for each developmental system. The actual configuration format can be found in [`config/developmental_systems.example.json`](config/developmental_systems.example.json); files such as `data/blood_features.json` and `data/neuron_features.json` referenced there are exported from the shared expression matrix via `ecasp prepare-rbp-expression`.

The current workflow directly uses a single matrix that already contains concatenated RBP + HVG features and has been standardized, for example [`data/tissue_expression_features_scaled.csv`](data/tissue_expression_features_scaled.csv). Rows are tissue names and columns are features. To generate `blood_features.json`, run:

```bash
ecasp prepare-rbp-expression \
  --matrix data/tissue_expression_features_scaled.csv \
  --tissue blood \
  --output data/blood_features.json \
  --format json \
  --standardize none
```

Output format:

```json
{
  "values": [...],
  "rbp_names": ["feature1", "feature2", ...]
}
```

The same applies to other systems. In addition, Stage 2 reference fine-tuning requires a neutral condition input. The repository already provides [`data/zero_rbp_features.json`](data/zero_rbp_features.json) and [`data/zero_rbp_features.npy`](data/zero_rbp_features.npy) as the zero vector for this step.

---

## Stage 1: Multi-System Joint Training (`train`)

After preparing condition vectors for each system, ECASP can be jointly trained across multiple developmental systems. Each system provides its own train/validation/test HDF5 files plus the corresponding condition vector. Training uses `--tissue-config` to mix mini-batches from different systems in a virtual mixed-batch scheme while sharing the same sequence backbone and FiLM branch.

See [`config/developmental_systems.example.json`](config/developmental_systems.example.json) for an example `--tissue-config`.

Example command:

```bash
ecasp train \
  --tissue-config config/developmental_systems.json \
  --flanking-size 10000 \
  --epochs 20 \
  --lr 1e-4 \
  --film-lr-mult 1.0 \
  --scheduler CosineAnnealingWarmRestarts \
  --loss focal_loss \
  --focal-alpha 0.2 1.0 1.0 \
  --focal-gamma 2.0 \
  --output-dir runs/stage1_multisystem \
  --project-name ecasp_stage1 \
  --random-seed 42
```

**Notes**:

- Every entry in `--tissue-config` must provide train/valid/test HDF5 files and the matching `rbp_expression`; all vectors must share the same dimensionality and feature order.
- FiLM is enabled in this stage, and `--film-lr-mult 1.0` means FiLM and backbone are trained together.
- To stay close to the Methods, use 15 developmental systems, 20 epochs, focal loss, and cosine warm restarts.
- The best checkpoint from Stage 1, `runs/stage1_multisystem/model_best.pt`, is used to initialize Stage 2 reference fine-tuning.

---

## Stage 2: Freeze FiLM and Fine-Tune the Backbone on Reference Data (`transfer`)

In the current Methods, Stage 2 starts from the best checkpoint of Stage 1 and continues training on the reference dataset. The two key points are: 1) the condition input is replaced with a zero vector so that this stage corresponds to a neutral condition; 2) the FiLM learning-rate multiplier is set to 0 so that only non-FiLM parameters are optimized, effectively freezing FiLM while fine-tuning the backbone/head.

Example command:

```bash
ecasp transfer \
  --train-dataset /path/reference/dataset_train.h5 \
  --test-dataset /path/reference/dataset_test.h5 \
  --pretrained-model runs/stage1_multisystem/model_best.pt \
  --flanking-size 10000 \
  --epochs 20 \
  --lr 1e-4 \
  --scheduler CosineAnnealingWarmRestarts \
  --rbp-expression data/zero_rbp_features.json \
  --film-lr-mult 0.0 \
  --unfreeze-all \
  --output-dir runs/stage2_reference \
  --project-name ecasp_stage2_reference
```

**Key notes**:

- `--pretrained-model` should point to the best checkpoint from Stage 1 multi-system joint training.
- `--rbp-expression data/zero_rbp_features.json` provides the zero-vector input required for reference fine-tuning and corresponds to the neutral condition described in the Methods.
- `--film-lr-mult 0.0` reduces the FiLM learning rate to zero and therefore freezes FiLM; `--unfreeze-all` allows all remaining non-FiLM parameters to continue training.
- As in `train`, the `train-dataset` filename only needs to contain `train`; the program automatically resolves `dataset_validation.h5` in the same directory.
- The final checkpoint from Stage 2, `runs/stage2_reference/model_best.pt`, is the recommended model for all downstream `predict` and `variant` runs.

The code still supports single-tissue transfer or other freezing strategies, but if you want the workflow to match the paper Methods, the preferred path is: Stage 1 multi-system joint training followed by Stage 2 reference fine-tuning with FiLM frozen.

---

## Variant Annotation (`variant`)

Finally, pass a VCF file to the system-specific model to obtain delta scores and splice-site shifts:

```bash
ecasp variant \
  --input data/decipher_variants_all.vcf \
  --output results/annotated_neuron.vcf \
  --model runs/stage2_reference/model_best.pt \
  --ref-genome data/genome.fa \
  --annotation data/grch38_chr.txt \
  --flanking-size 10000 \
  --rbp-expression data/neuron_features.json
```

**Notes**:

- If the checkpoint contains FiLM/conditioning metadata, `variant` validates the dimensionality and feature names of the input vector. Missing or misordered inputs will raise an error to avoid silent prediction bias.
- To reproduce the paper workflow, always use the final Stage 2 checkpoint and only switch `--rbp-expression` to move between developmental systems. Because the checkpoint is fixed, scores remain directly comparable across systems.

---

## Sequence-Level Prediction (`predict`)

`predict` supports RBP/HVG condition vectors and can directly output tissue-specific splice-site BED files from FASTA input. Example:

```bash
ecasp predict \
  --input-sequence data/neuron_genes.fa \
  --model runs/stage2_reference/model_best.pt \
  --flanking-size 10000 \
  --rbp-expression data/neuron_features.json \
  --output-dir predict_out/neuron/ \
  --threshold 1e-6 \
  --predict-all
```

**Notes**:

- `--rbp-expression` follows the same rule as in `variant`: it must match the `rbp_dim` and `rbp_names` recorded in the FiLM checkpoint. Missing or misordered vectors will raise an error.
- To reproduce the paper workflow, use the final Stage 2 checkpoint and provide the real expression vector of the target developmental system during inference.
- `--predict-all` first writes intermediate HDF5/pt files and then exports BED. Without it, BED files are written incrementally to save disk space.
- The resulting `acceptor_predictions.bed` and `donor_predictions.bed` can be compared across systems such as limb vs neuron.

---

## RBP Gradient / Contribution Analysis

If you want to inspect conditional-input gradients for a single variant in a given developmental-system context, use `ecasp gradient-rbp-attribution`. Its implementation lives in [`ecasp/scripts/gradient_rbp_attribution.py`](ecasp/scripts/gradient_rbp_attribution.py). The command below computes gradients for RBP/HVG features and writes the result to a TSV file:

```bash
ecasp gradient-rbp-attribution \
  --variant 'chr17:28369751:G>GC' \
  --gene VTN \
  --model runs/stage2_reference/model_best.pt \
  --ref-genome /path/genome.fa \
  --annotation data/grch38_chr.txt \
  --flanking-size 10000 \
  --rbp-expression data/blood_features.json \
  --film-strengths 1 \
  --target DS_AG \
  --rank-metric grad \
  --top-k 20 \
  --output results/vtn_blood_rbp_gradient.tsv
```

Notes:

- `--rank-metric grad` ranks by raw gradient; to rank by the paper-style contribution score, switch back to the default `--rank-metric gradxinput`.
- The output includes both `grad` and `grad_x_input`, so one run lets you inspect raw gradients and contribution-style values together.
- If you already have a curated RBP list, add `--rbp-only --rbp-list /path/all_RBP_gene_names.txt` to restrict the output to RBP features only.
