







  


ECASP (Expression-Conditioned AI for Splicing Prediction) is an open‐source, efficient, and modular framework for splice site prediction. It builds on the OpenSpliceAI/SpliceAI lineage with a modern PyTorch implementation and provides researchers with a user‐friendly suite of tools for creating training datasets, training models, predicting splice sites, and assessing the impact of genetic variants.

  


# Key Features[#](#key-features)

- **Modern, Retrainable Framework:** Built on Python 3 and PyTorch, ECASP improves the limitations of older TensorFlow/Keras implementations. Its modular design enables fast and efficient prediction, as well as easy retraining on species-specific data with just a few commands.
- **Updated and Cross-Species Models:** ECASP includes a pre-trained human model, **OSAIMANE-10000nt**, updated from GRCh37 to GRCh38 using the latest MANE annotations, along with models for mouse, thale cress (*Arabidopsis*), honey bee, and zebrafish. This versatility empowers researchers to study splicing across diverse species.
- **Variant Impact Prediction:** ECASP not only predicts splice sites but also assesses the impact of genetic variants (SNPs and INDELs) on splicing. Its `variant` subcommand calculates “delta” scores that quantify changes in splice site strength and predicts cryptic splice sites.
- **Efficiency and Scalability:** Optimized for improved processing speeds, lower memory usage, and efficient GPU utilization, ECASP can handle large genomic regions and whole-genome predictions on a single GPU.

  


# Who Should Use ECASP?[#](#who-should-use-openspliceai)

- **Human Genomics Researchers:** Use the newly retrained ECASP model, **OSAIMANE-10000nt**, for highly accurate splice site predictions based on the latest human annotations.
- **Comparative and Non-Human Genomics:** Whether you’re studying mouse, zebrafish, honey bee, or thale cress, ECASP offers models pre-trained on multiple species — and the ability to train your own models — ensuring broad applicability.
- **Variant Analysts:** If you need to predict how genetic variants affect splicing, ECASP’s variant subcommand provides detailed delta scores and positional information to assess functional impacts.

  


# What ECASP Does[#](#what-openspliceai-does)

- **Data Preprocessing** ([create-data](https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#create-data-subcommand)): Converts genome FASTA and annotation (GFF/GTF) files into one-hot encoded datasets (HDF5 format) for training and testing.
- **Model Training** ([train](https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#train-subcommand)): Trains deep residual convolutional neural networks on the preprocessed datasets. ECASP supports training from scratch and employs adaptive learning rate schedulers and early stopping.
- **Transfer Learning** ([transfer](https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#transfer-subcommand)): Fine-tunes a pre-trained human model for other species, reducing training time and improving performance on species with limited data.
- **Model Calibration** ([calibrate](https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#calibrate-subcommand)): Adjusts model output probabilities to better reflect true splice site likelihoods, enhancing prediction accuracy.
- **Prediction** ([predict](https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#predict-subcommand)): Uses trained models to generate splice site predictions from FASTA sequences, outputting BED files with donor and acceptor site coordinates.
- **Variant Analysis** ([variant](https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#variant-subcommand)): Annotates VCF files with delta scores and positions to evaluate the impact of genetic variants on splicing.

  


# RBP-Conditioned Prediction[#](#rbp-conditioned-prediction)

ECASP now includes optional FiLM (Feature-wise Linear Modulation) layers so that later residual units can be conditioned on a fixed RNA binding protein (RBP) expression vector. This keeps the early layers focused on universal sequence rules while the final layers adapt to a tissue-specific regulatory environment.

1. **Prepare a tissue feature vector.** The helper script now concatenates RBP expression (`data/tissue_rbp_matrix.csv`) with optional high-variance gene (HVG) profiles (e.g. `data/developmental_system_hvg.csv`).

```
python -m openspliceai.scripts.prepare_rbp_expression \
  --matrix data/tissue_rbp_matrix.csv \
  --hvg-matrix data/developmental_system_hvg.csv \
  --tissue limb \
  --output data/limb_rbp.json \
  --format json \
  --standardize zscore \
  --hvg-standardize zscore

```

The resulting JSON/NPY stores a single concatenated vector whose `rbp_names` entry reflects both the RBP and HVG feature names; the same file must be reused during training and inference.

1. **Condition transfer learning.** Pass `--rbp-expression` to enable FiLM; the tissue vector modulates the tail feature map immediately before the final 1×1 head:

```
ecasp transfer \
  --train-dataset /home1/xyf/data/openspliceai_data/dataset_limb/train.h5 \
  --test-dataset /home1/xyf/data/openspliceai_data/dataset_limb/test.h5 \
  --pretrained-model data/model_best.pt \
  --rbp-expression data/limb_rbp.json \
  --output-dir runs/limb_rbp \
  --project-name limb_rbp

```

*Note:* The `train` subcommand also accepts `--tissue-config` for multi-tissue joint training from scratch (no `--pretrained-model` required).

1. **Run RBP-aware variant annotation.** Tissue-specific checkpoints store their required RBP dimensionality. The `variant` subcommand validates that the same vector is provided at inference time:

```
ecasp variant \
  --input decipher_variants_all.vcf \
  --output results/annotated.vcf \
  --model model_limb.pt \
  --ref-genome data/genome.fa \
  --annotation data/grch38.txt \
  --rbp-expression data/limb_rbp.json

```

**Tip:** The checkpoint records the RBP dimensionality (and any supplied names), so mismatched or missing vectors are reported immediately. Base models without FiLM layers continue to work without any additional inputs.

# User Support & Contributors[#](#user-support-contributors)

If you have questions, encounter issues, or would like to request a new feature, please use our GitHub issue tracker at: [https://github.com/Kuanhao-Chao/OpenSpliceAI/issues](https://github.com/Kuanhao-Chao/OpenSpliceAI/issues)

ECASP was developed by Kuan-Hao Chao, Alan Mao, and collaborators at Johns Hopkins University. For further details on usage, methods, and performance, please refer to the full documentation and online methods sections.

  


# Next Steps[#](#next-steps)

Check out the [Installation Guide](https://ccb.jhu.edu/openspliceai/content/installation.html#installation) to get started with ECASP. For a quick overview of the main commands and subcommands, see the [Quick Start Guide](https://ccb.jhu.edu/openspliceai/content/quick_start_guide/index.html#quick-start-home).

  


# Table of Contents[#](#table-of-contents)

- [Installation](https://ccb.jhu.edu/openspliceai/content/installation.html)
  - [Overview](https://ccb.jhu.edu/openspliceai/content/installation.html#overview)
  - [Prerequisites](https://ccb.jhu.edu/openspliceai/content/installation.html#prerequisites)
  - [Installation Methods](https://ccb.jhu.edu/openspliceai/content/installation.html#installation-methods)
  - [Detailed Installation for PyTorch and mappy](https://ccb.jhu.edu/openspliceai/content/installation.html#detailed-installation-for-pytorch-and-mappy)
  - [Check ECASP Installation](https://ccb.jhu.edu/openspliceai/content/installation.html#check-openspliceai-installation)
  - [Terminal Output Example](https://ccb.jhu.edu/openspliceai/content/installation.html#terminal-output-example)
  - [Next Steps](https://ccb.jhu.edu/openspliceai/content/installation.html#next-steps)
- [Quick Start Guide](https://ccb.jhu.edu/openspliceai/content/quick_start_guide/index.html)
  - [Usage 1 – Predict](https://ccb.jhu.edu/openspliceai/content/quick_start_guide/index.html#usage-1-predict)
  - [Usage 2 – Train from Scratch](https://ccb.jhu.edu/openspliceai/content/quick_start_guide/index.html#usage-2-train-from-scratch)
  - [Usage 3 – Transfer Learning](https://ccb.jhu.edu/openspliceai/content/quick_start_guide/index.html#usage-3-transfer-learning)
- [create-data](https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html)
  - [Input Files](https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#input-files)
  - [Output Files](https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#output-files)
  - [Usage](https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#usage)
  - [Examples](https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#examples)
  - [Processing Pipeline](https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#processing-pipeline)
  - [Conclusion](https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#conclusion)
- [train](https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html)
  - [Subcommand Description](https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#subcommand-description)
  - [Input Files](https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#input-files)
  - [Output Files](https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#output-files)
  - [Usage](https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#usage)
  - [Examples](https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#examples)
  - [Processing Steps](https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#processing-steps)
  - [Conclusion](https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#conclusion)
- [transfer](https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html)
  - [Subcommand Description](https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#subcommand-description)
  - [Input Files](https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#input-files)
  - [Output Files](https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#output-files)
  - [Usage](https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#usage)
  - [Examples](https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#examples)
  - [Processing Pipeline](https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#processing-pipeline)
  - [Conclusion](https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#conclusion)
- [calibrate](https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html)
  - [Subcommand Description](https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#subcommand-description)
  - [Input Files](https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#input-files)
  - [Output Files](https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#output-files)
  - [Usage](https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#usage)
  - [Examples](https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#examples)
  - [Processing Steps](https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#processing-steps)
  - [Conclusion](https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#conclusion)
- [predict](https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html)
  - [Overview](https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#overview)
  - [Input Files](https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#input-files)
  - [Output Files](https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#output-files)
  - [Usage](https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#usage)
  - [Examples](https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#examples)
  - [Processing Pipeline](https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#processing-pipeline)
  - [Conclusion](https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#conclusion)
- [variant](https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html)
  - [Overview](https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#overview)
  - [Input Files](https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#input-files)
  - [Output Files](https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#output-files)
  - [Delta Score Computation](https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#delta-score-computation)
  - [Usage](https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#usage)
  - [Examples](https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#examples)
  - [Processing Pipeline](https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#processing-pipeline)
  - [Conclusion](https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#conclusion)
- [Steps & Commands to Train ECASP Models](https://ccb.jhu.edu/openspliceai/content/train_your_own_model/index.html)
- [Released ECASP models](https://ccb.jhu.edu/openspliceai/content/pretrained_models/index.html)
- [ECASP vs. SpliceAI](https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html)
  - [Overview](https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html#overview)
  - [Architectural and Implementation Differences](https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html#architectural-and-implementation-differences)
  - [Performance and Efficiency](https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html#performance-and-efficiency)
  - [Training Flexibility and Transfer Learning](https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html#training-flexibility-and-transfer-learning)
  - [Model Calibration and Variant Analysis](https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html#model-calibration-and-variant-analysis)
  - [Conclusion](https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html#conclusion)
- [Behind the Scenes](https://ccb.jhu.edu/openspliceai/content/behind_scenes.html)
  - [Architecture and Framework](https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#architecture-and-framework)
  - [Data Preprocessing and One-Hot Encoding](https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#data-preprocessing-and-one-hot-encoding)
  - [Training and Optimization](https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#training-and-optimization)
  - [Transfer Learning](https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#transfer-learning)
  - [Model Calibration](https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#model-calibration)
  - [Variant Effect Analysis](https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#variant-effect-analysis)
  - [Performance and Benchmarking](https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#performance-and-benchmarking)
  - [Conclusion](https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#conclusion)
- [Q & A](https://ccb.jhu.edu/openspliceai/content/how_to_page.html)
- [Changelog](https://ccb.jhu.edu/openspliceai/content/changelog.html)
  - [v1.0.0](https://ccb.jhu.edu/openspliceai/content/changelog.html#v1-0-0)
- [License](https://ccb.jhu.edu/openspliceai/content/license.html)
- [Contact](https://ccb.jhu.edu/openspliceai/content/contact.html)

  


  


  


  


  
