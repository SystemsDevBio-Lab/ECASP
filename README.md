# ECASP Workflow Guide

Project logo (PDF): [Logo.pdf](Logo.pdf)

ECASP (Expression-Conditioned AI for Splicing Prediction) is a PyTorch-based splice prediction framework that reconstructs and extends the SpliceAI workflow from raw genome inputs to splice-site prediction and variant annotation. This guide presents the full workflow from "zero to a tissue-specific model", covering data preparation, base-model training, FiLM-conditioned fine-tuning, and variant annotation so you can reproduce the pipeline on your own species or tissue.

---

## 1. Environment and Resources

- **Dependencies**: Python >= 3.10, PyTorch (CUDA >= 11.7 recommended for GPU training), NumPy/Pandas/HDF5/pyfaidx, and related packages. Running `pip install -e .` installs the required Python dependencies.
- **Hardware**: At least 16 GB GPU memory is recommended for base-model training and FiLM fine-tuning. Variant annotation can run on CPU, but it will be slower.
- **Core inputs**:
  - Reference genome FASTA (for example `data/genome.fa` plus `.fai`).
  - Tissue- or species-specific GTF/GFF annotation files.
  - SpliceAI annotation files (for example `data/grch38.txt`) or a custom annotation.
  - A standardized tissue expression matrix containing concatenated RBP + HVG features, such as `data/tissue_expression_features_scaled.csv`.

After installation, you can verify the CLI with:

```bash
pip install -e .
ecasp --help
```

---

## 2. Full Workflow Overview

1. **create-data**: Read GTF/GFF and FASTA files and generate HDF5 datasets for training, validation, and testing.
2. **train**: Train a base model on the generated HDF5 datasets without FiLM to learn general splice rules.
3. **prepare_rbp_expression**: Build a tissue condition vector containing RBP + HVG features.
4. **transfer**: Load a base-model checkpoint, enable FiLM, and fine-tune on tissue-specific data.
5. **predict**: Use a tissue-specific model plus a condition vector to predict splice sites from FASTA input and output BED files.
6. **variant**: Use a tissue-specific model plus a condition vector to annotate VCF variants for predicted splicing effects.

The sections below describe each step in detail.

---

## 3. Step 1: Generate HDF5 Data (`create-data`)

`create-data` performs two tasks: (1) `create_datafile` slices annotation data into sequence windows; (2) `create_dataset` exports HDF5 shards. A typical command is:

```bash
ecasp create-data \
  --annotation-gff data/limb_filtered.gff3 \
  --genome-fasta data/genome.fa \
  --output-dir /home1/xyf/data/openspliceai_data/dataset_limb \
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

**Key points**

- `--flanking-size` controls the input sequence length. Longer inputs capture more distal context but require more memory, especially at 10000 nt.
- `--chr-split` and `--split-method` define the chromosome splitting strategy. `human` follows the split convention used in the original SpliceAI paper.
- `--remove-paralogs` uses minimap2 to detect homologous sequences between training and test sets and reduce leakage.
- `--verify-h5` runs a consistency check after HDF5 generation.

The output directory usually contains `dataset_train.h5`, `dataset_validation.h5`, `dataset_test.h5`, plus logs and summary files. These files are used directly by later `train` and `transfer` steps, and the filenames should retain the `train` / `validation` / `test` keywords.

> **Tip: preprocessing tissue-specific GFF3 files**  
> `workflow/step_all.sh` chains `step0` through `step3` and can generate `<tissue>_step3.gff3` with `bash workflow/step_all.sh <tissue>`. Each step filters isoforms, converts GFF3, renames transcript to mRNA, and adds `gene_biotype`, producing files ready for `create-data`.

---

## 4. Step 2: Train the Base Model (`train`)

The base model learns tissue-agnostic splice rules. FiLM fine-tuning later injects tissue-specific information on top of this base. Example command:

```bash
ecasp train \
  --train-dataset /home1/xyf/data/openspliceai_data/dataset_limb/dataset_train.h5 \
  --test-dataset /home1/xyf/data/openspliceai_data/dataset_limb/dataset_test.h5 \
  --flanking-size 10000 \
  --epochs 10 \
  --scheduler CosineAnnealingWarmRestarts \
  --loss cross_entropy_loss \
  --output-dir runs/base_model \
  --project-name limb_base \
  --random-seed 42
```

**Notes**

- The `train-dataset` filename must contain `train`; the program automatically swaps the same directory filename to `validation` to load the validation split.
- The `train` mode does not enable FiLM by default and does not require RBP/HVG features.
- The output directory contains epoch checkpoints (`model_{epoch}.pt`), the best checkpoint, and training/validation logs such as AUPRC and loss curves.
- If you train multiple species, switching the HDF5 inputs in `create-data` and `train` produces separate base models.

---

## 5. Step 3: Build a Tissue Condition Vector (`prepare_rbp_expression`)

FiLM requires a fixed condition vector. The current workflow uses a single standardized matrix that already includes both RBP and HVG features, for example `data/tissue_expression_features_scaled.csv`, where rows are tissues and columns are features. Run:

```bash
python -m openspliceai.scripts.prepare_rbp_expression \
  --matrix /home1/xyf/project/github/OpenSpliceAI/data/tissue_expression_features_scaled.csv \
  --tissue limb \
  --output data/limb_features.json \
  --format json \
  --standardize none
```

Output format:

```json
{
  "values": [...],
  "rbp_names": ["feature1", "feature2", "..."]
}
```

**Reuse the same condition file during both training and inference.** The checkpoint stores `rbp_dim` and `rbp_names` and uses them for validation. If you have additional tissue-level features, append them as columns to the matrix and export them with the same script.

---

## 6. Step 4: FiLM Fine-Tuning (`transfer`)

`transfer` loads a base model and attaches a FiLM side MLP that injects the tissue vector at the end of the backbone, after the residual stack and before the final 1x1 convolution. Depending on dataset size, you can use either **single-tissue mode** or **multi-tissue shared mode**.

### 6.1 Single-Tissue Mode

```bash
ecasp transfer \
  --train-dataset /home1/xyf/data/openspliceai_data/dataset_limb/dataset_train.h5 \
  --test-dataset /home1/xyf/data/openspliceai_data/dataset_limb/dataset_test.h5 \
  --pretrained-model runs/base_model/model_best.pt \
  --flanking-size 10000 \
  --epochs 5 \
  --rbp-expression data/limb_features.json \
  --unfreeze 4 \
  --output-dir runs/limb_film \
  --project-name limb_film
```

Important details for the updated FiLM workflow:

- As in `train`, the `train-dataset` filename only needs to include `train`; the program automatically resolves the matching `dataset_validation.h5`.
- `--rbp-expression` points to the JSON/NPY exported in the previous step and contains concatenated RBP + HVG features.
- FiLM is injected at a fixed point near the model tail, immediately before the final 1x1 head.
- The FiLM side MLP is `Linear(in_dim->128) -> LayerNorm -> ReLU -> Dropout(0.2) -> Linear(128->2*channels)`, with zero-initialized output weights and bias so the starting state is gamma = 1 and beta = 0.
- Training freezes the backbone by default, while always unfreezing the FiLM branch and final 1x1 head. `--unfreeze` additionally unfreezes the last few residual units; `--unfreeze-all` enables full fine-tuning.
- `model_best.pt` records FiLM metadata such as `rbp_dim` and `rbp_names`. If `--rbp-expression` is omitted, FiLM falls back to gamma = 1 and beta = 0 and behaves like the base model.

### 6.2 Multi-Tissue Shared Mode: `--tissue-config`

If you want a single FiLM checkpoint that already sees multiple tissues during training, provide a JSON config describing all tissue HDF5 files and condition vectors:

```json
[
  {
    "name": "blood",
    "train_dataset": "/path/blood/dataset_train.h5",
    "valid_dataset": "/path/blood/dataset_validation.h5",
    "test_dataset": "/path/blood/dataset_test.h5",
    "rbp_expression": "data/blood_features.json"
  },
  {
    "name": "neuron",
    "train_dataset": "/path/neuron/dataset_train.h5",
    "valid_dataset": "/path/neuron/dataset_validation.h5",
    "test_dataset": "/path/neuron/dataset_test.h5",
    "rbp_expression": "data/neuron_features.json"
  }
]
```

Example command:

```bash
ecasp transfer \
  --tissue-config config/tissues.json \
  --pretrained-model runs/base_model/model_best.pt \
  --flanking-size 10000 \
  --epochs 5 \
  --unfreeze 4 \
  --output-dir runs/shared_film \
  --project-name shared_film
```

Notes:

- `--tissue-config` and `--rbp-expression` are mutually exclusive. The config-based mode loads multiple tissues in one run, and each tissue must provide train/valid/test HDF5 files.
- All `rbp_expression` vectors in the config must have identical dimensionality and feature ordering.
- During training, the dataloader mixes batches from different tissues. The same FiLM side MLP generates gamma/beta, conditioned on the matching tissue vector.
- After training, the Variant step only needs this single checkpoint. You can run `ecasp variant` multiple times with different `--rbp-expression` files such as blood and neuron to obtain comparable tissue-specific predictions.
- Multi-tissue mode no longer auto-rescales batch size. It uses the single-tissue baseline batch size, with gradient accumulation defaulting to the number of tissues.
- `train` also supports `--tissue-config` for multi-tissue joint training from scratch, without `--pretrained-model`.

Example multi-tissue `train` run:

```bash
ecasp train \
  --tissue-config config/tissues.json \
  --flanking-size 400 \
  --epochs 8 \
  --lr 1e-3 \
  --film-lr-mult 1.0 \
  --loss focal_loss \
  --focal-alpha 0.25 0.25 0.5 \
  --focal-gamma 2.0 \
  --output-dir runs/shared_film_train \
  --project-name shared_film_train \
  --early-stopping --patience 2
```

---

## 7. Step 5: Sequence-Level Prediction (`predict`)

`predict` supports RBP/HVG condition vectors and can generate tissue-specific splice-site BED outputs directly from FASTA input. Example:

```bash
ecasp predict \
  --input-sequence data/neuron_genes.fa \
  --model runs/shared_film/model_best.pt \
  --flanking-size 10000 \
  --rbp-expression data/neuron_features.json \
  --output-dir predict_out/neuron/ \
  --threshold 1e-6 \
  --predict-all
```

Notes:

- `--rbp-expression` must match the `rbp_dim` and `rbp_names` stored in the FiLM checkpoint. Missing or mismatched vectors trigger an error.
- If you load a base model without FiLM, you can omit `--rbp-expression`; the behavior then matches standard SpliceAI.
- `--predict-all` first writes intermediate HDF5/pt outputs and then generates BED files. Without it, the tool writes BED incrementally to save disk space.
- The resulting `acceptor_predictions.bed` and `donor_predictions.bed` can be compared across tissues such as limb vs neuron.

---

## 8. Step 6: Variant Annotation (`variant`)

Finally, annotate VCF variants with a tissue-specific model:

```bash
ecasp variant \
  --input data/decipher_variants_all.vcf \
  --output results/annotated_limb.vcf \
  --model runs/limb_film/model_best.pt \
  --ref-genome data/genome.fa \
  --annotation data/grch38_chr.txt \
  --flanking-size 10000 \
  --rbp-expression data/limb_features.json
```

**Notes**

- If the checkpoint contains FiLM metadata, `variant` validates both the dimensionality and the feature names of the supplied vector.
- For multi-tissue shared models, simply rerun `variant` with different `--rbp-expression` files to generate blood, neuron, or other tissue-specific outputs. Since the checkpoint is the same, the resulting scores are directly comparable.

---

## 9. Example Outputs and Directory Structure

```text
ECASP/
├── data/
│   ├── genome.fa / genome.fa.fai
│   ├── grch38.txt
│   ├── tissue_rbp_matrix.csv
│   └── developmental_system_hvg.csv
├── runs/
│   ├── base_model/
│   │   ├── model_best.pt
│   │   └── metrics/*.txt
│   └── limb_film/
│       ├── model_best.pt
│       └── metrics/*.txt
├── results/
│   └── annotated_limb.vcf
└── scripts/prepare_rbp_expression.py
```

It is good practice to archive training logs and checkpoints together so different tissues and parameter settings can be compared later.

---

## 10. FAQ

1. **Can a FiLM model run without a condition vector?**  
   Yes. If `--rbp-expression` is not provided, FiLM falls back to gamma = 1 and beta = 0, which is equivalent to the standard SpliceAI behavior. The program only errors when the checkpoint explicitly requires a nonzero `rbp_dim` and no vector is supplied.

2. **How can I customize tissue features?**  
   Concatenate any tissue-level features such as RBP TPM, highly variable gene expression, or UMAP coordinates into a CSV whose row names match `tissue_rbp_matrix.csv`. Then pass it through `--hvg-matrix` or replace the original matrix directly. `prepare_rbp_expression` will handle concatenation and standardization.

3. **Can one training run cover multiple tissues?**  
   Yes. Use `--tissue-config` to provide multiple tissues with train/valid/test HDF5 files and matching condition vectors. `transfer` then performs mixed training with a shared FiLM branch.

4. **How should I interpret the Variant output?**  
   `variant` produces the same delta score set as official SpliceAI: Delta AG, AL, DG, and DL, together with position offsets. These values can be used to prioritize potentially splice-altering variants. If you run the same variants across multiple tissues, their score differences become directly comparable.

---

## 11. Further Resources

- Documentation and tutorials: `README.md` and the `docs/` directory.
- Relevant code paths:
  - `openspliceai/create_data/*`: HDF5 generation and validation
  - `openspliceai/train_base/*`: SpliceAI backbone, FiLM layers, and training/validation loops
  - `openspliceai/rbp/expression.py`: condition-vector I/O and standardization
  - `openspliceai/variant/variant.py`: VCF annotation entry point
- If you encounter issues or want to contribute improvements, please open a GitHub issue.

We hope ECASP is useful for your splicing research.
