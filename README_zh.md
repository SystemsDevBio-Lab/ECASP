<p align="center">
  <img src="Logo.png" alt="ECASP logo" width="720">
</p>

ECASP（Expression-Conditioned AI for Splicing Prediction）是在 PyTorch 中重构和扩展的剪接预测框架，实现了从原始基因组数据到剪接位点预测/变异注释的全流程。本指南按当前 methods 中采用的两阶段训练主线组织：先在多个 developmental systems 上联合训练 ECASP（保留 FiLM 条件分支），再冻结 FiLM，使用 reference dataset 在零向量条件下微调 sequence backbone；最终在推理和 variant 注释阶段重新输入目标 system 的真实表达向量，从而得到 system-specific prediction。

---

## 1. 环境与资源

- **依赖**：Python ≥ 3.10、PyTorch（GPU 训练建议 CUDA≥11.7）、NumPy/Pandas/HDF5/pyfaidx 等；执行 `pip install -e .` 会自动安装需要的 Python 包。
- **硬件**：基础模型/FiLM 微调建议使用至少 16GB GPU；Variant 注释可在 CPU 上运行但会较慢。
- **基础数据**：
  - 参考基因组 FASTA（例：`data/genome.fa` + `.fai`）。
  - 组织/物种对应的 GTF/GFF 注释文件。
  - SpliceAI 官方 annotation（示例：`data/grch38.txt`）或自定义注释。
  - 组织表达矩阵：已拼接且标准化的 RBP+HVG 矩阵，如 `data/tissue_expression_features_scaled.csv`。

安装完成后可用下列命令快速检查：

```bash
pip install -e .
ecasp --help
```

---

## 2. 完整流程概览

1. **create-data**：分别准备 reference dataset 和多 developmental systems 的 HDF5 数据集（训练/验证/测试）。
2. **prepare-rbp-expression**：为每个 developmental system 生成条件向量（RBP + HVG），并准备 reference fine-tuning 所需的零向量。
3. **train**：第一阶段，在多个 developmental systems 上联合训练 ECASP，FiLM 与 backbone 一起学习。
4. **transfer**：第二阶段，以第一阶段最佳 checkpoint 为起点，冻结 FiLM，在 reference dataset 上微调非 FiLM 参数。
5. **predict**：使用第二阶段最终模型 + 目标 system 条件向量，对 FASTA 序列做 system-specific 剪接位点预测（输出 BED）。
6. **variant**：使用第二阶段最终模型 + 目标 system 条件向量，对 VCF 做 system-specific 剪接影响注释。

以下章节将详细说明每一步。

---

## 3. Step 1：生成 HDF5 数据 (`create-data`)

`create-data` 会执行两件事：1) `create_datafile` 将注释切成序列窗口；2) `create_dataset` 输出 HDF5 分片。典型命令：

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
我们生成的多组织训练数据见（xxxxxxxx），reference dataset 见（xxxxxxxxx）
---

## 4. Step 2：生成条件向量 (`prepare-rbp-expression`)

在进入第一阶段多组织联合训练之前，需要先为每个 developmental system 准备条件向量。`config/developmental_systems.json` 中引用的 `data/blood_features.json`、`data/neuron_features.json` 这类文件，都是通过 `ecasp prepare-rbp-expression` 从共享表达矩阵导出的。

现在直接使用单个矩阵（已包含 RBP + HVG，且已标准化）例如 `data/tissue_expression_features_scaled.csv`，行是组织名、列是特征名。以 `blood_features.json` 为例，运行：

```bash
ecasp prepare-rbp-expression \
  --matrix data/tissue_expression_features_scaled.csv \
  --tissue blood \
  --output data/blood_features.json \
  --format json \
  --standardize none
```

输出文件格式：

```json
{
  "values": [...],          # RBP + HVG 向量
  "rbp_names": ["feature1", "feature2", ...]
}
```

其他系统同理，只需替换 `--tissue` 和 `--output`，例如生成 neuron 向量：

```bash
ecasp prepare-rbp-expression \
  --matrix data/tissue_expression_features_scaled.csv \
  --tissue neuron \
  --output data/neuron_features.json \
  --format json \
  --standardize none
```

**务必在训练和推理时复用同一文件**，checkpoint 会记录 `rbp_dim` 与 `rbp_names` 用于校验。如果你有额外的组织特征，可直接追加到该矩阵列中，再用本脚本导出。

此外，第二阶段 reference fine-tuning 需要“中性条件”输入。仓库已提供 `data/zero_rbp_features.json` / `data/zero_rbp_features.npy` 作为零向量，可直接用于这一阶段。

---

## 5. Step 3：第一阶段，多组织联合训练 (`train`)

完成各 system 的条件向量准备后，即可在多个 developmental systems 上联合训练 ECASP。每个 system 提供独立的 train/validation/test HDF5 与对应条件向量；训练时通过 `--tissue-config` 以 virtual mixed-batch 方式混合各 system 的 mini-batch，并共享同一套 sequence backbone 与 FiLM 分支。

`config/developmental_systems.json` 示例：

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

示例命令：

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

**提示**：

- `--tissue-config` 中每个条目都要提供 train/valid/test HDF5 与对应 `rbp_expression`；所有向量的维度与特征顺序必须一致。
- 该阶段 FiLM 处于启用状态，`--film-lr-mult 1.0` 表示 FiLM 与 backbone 一起训练。
- 若要贴近 methods，可使用 15 个 developmental systems、20 个 epoch、focal loss 和 cosine warm restart。
- 第一阶段输出的 `runs/stage1_multisystem/model_best.pt` 是第二阶段 reference fine-tuning 的初始化 checkpoint。

---

## 6. Step 4：第二阶段，冻结 FiLM 并用 reference data 微调 backbone (`transfer`)

当前 methods 中的第二阶段以上一步最佳 checkpoint 为起点，在 reference dataset 上继续训练。关键点有两个：1）条件输入改为零向量，使这一阶段对应“中性条件”；2）FiLM 分支学习率乘子设为 0，只更新非 FiLM 参数，从而实现“冻结 FiLM、微调 backbone/head”。

示例命令：

```bash
ecasp transfer \
  --train-dataset /home1/xyf/data/openspliceai_data/reference/dataset_train.h5 \
  --test-dataset /home1/xyf/data/openspliceai_data/reference/dataset_test.h5 \
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

**关键说明**：

- `--pretrained-model` 使用第一阶段多组织联合训练得到的最佳 checkpoint。
- `--rbp-expression data/zero_rbp_features.json` 提供 reference fine-tuning 所需的零向量输入，对应 methods 中的 neutral condition。
- `--film-lr-mult 0.0` 会将 FiLM 分支学习率降为 0，从而冻结 FiLM；`--unfreeze-all` 允许其余非 FiLM 参数继续训练。
- 与 `train` 相同，`train-dataset` 名称中包含 `train` 即可，程序会自动在同目录定位 `dataset_validation.h5`。
- 第二阶段输出的 `runs/stage2_reference/model_best.pt` 才是后续 `predict` 和 `variant` 推荐使用的最终模型。

代码层面仍支持单组织 transfer 或其他冻结策略，但若要与论文 methods 保持一致，应优先采用“第一阶段多组织联合训练 + 第二阶段冻结 FiLM、reference fine-tuning”这条主线。

---

## 7. Step 5：序列级预测 (`predict`)

`predict` 现在支持 RBP/HVG 条件向量，可直接对 FASTA 序列输出组织特异的剪接位点 BED。示例：

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

说明与注意：

- `--rbp-expression` 与 `variant` 用法一致，必须与 FiLM checkpoint 中记录的 `rbp_dim`/`rbp_names` 对齐；缺失时会报错，维度或顺序不匹配也会报错。
- 若要复现本文工作流，请使用第二阶段输出的最终 checkpoint，并在推理时传入目标 developmental system 的真实表达向量。
- `--predict-all` 会先写中间 HDF5/pt，再生成 BED；关闭该选项则直接边预测边写 BED，节省磁盘。
- 输出的 `acceptor_predictions.bed`、`donor_predictions.bed` 可按组织对比（如 limb vs neuron）。

---

## 8. Step 6：Variant 注释 (`variant`)

最后，将 VCF 输入组织特异模型即可得到 delta 分数和剪接位点位移：

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

**注意**：

- 若 checkpoint 包含 FiLM/条件元数据，`variant` 会检查输入向量维度与名称；缺失或顺序错误会直接报错，避免预测偏差。
- 复现本文分析时，推荐始终使用第二阶段最终 checkpoint，仅替换 `--rbp-expression` 以切换不同 developmental system；由于 checkpoint 相同，分数可直接跨系统比较。

---

## 9. 结果与目录结构示例

```
ECASP/
├── data/
│   ├── genome.fa / genome.fa.fai
│   ├── grch38.txt
│   ├── tissue_rbp_matrix.csv
│   └── developmental_system_hvg.csv
├── runs/
│   ├── stage1_multisystem/
│   │   ├── model_best.pt
│   │   └── metrics/*.txt
│   └── stage2_reference/
│       ├── model_best.pt
│       └── metrics/*.txt
├── results/
│   └── annotated_neuron.vcf
└── scripts/prepare_rbp_expression.py
```

建议将训练日志（TensorBoard/自定义可视化）与 checkpoint 同步归档，方便比较不同组织或参数设置。

---

## 10. 常见问题

1. **FiLM 模型可以在没有条件向量时运行吗？**  
   - 可以；若不传 `--rbp-expression`，FiLM γ=1、β=0，相当于标准 SpliceAI。仅当 checkpoint 中声明 `rbp_dim>0` 且你忘记提供向量时，程序才会报错。

2. **如何自定义组织特征？**  
   - 将任意组织级别特征（如 RBP TPM、高变基因表达、UMAP 坐标等）拼成 CSV，行名与 `tissue_rbp_matrix.csv` 保持一致，再传给 `--hvg-matrix` 或直接替换原矩阵。`ecasp prepare-rbp-expression` 会自动拼接、标准化。

3. **一次训练能覆盖多个组织吗？**  
   - 可以。论文 methods 的第一阶段就是使用 `--tissue-config` 在多个 developmental systems 上联合训练共享的 ECASP 模型；第二阶段再固定 FiLM，用 reference dataset 微调非 FiLM 参数。

4. **Variant 结果如何解释？**  
   - `variant` 输出与官方 SpliceAI 相同的 delta scores（ΔAG、ΔAL、ΔDG、ΔDL）及位置偏移，可直接用于筛选可能影响剪接的突变。若对多个组织运行，可比较不同组织的分数差异。

---

## 11. 更多资源

- 文档/教程（英文）：`README.md` 与 `docs/` 目录。
- 相关脚本：
  - `openspliceai/create_data/*`：HDF5 生成与验证。
  - `openspliceai/train/*` 与 `openspliceai/transfer/*`：两阶段训练与 fine-tuning 入口。
  - `openspliceai/scripts/prepare_rbp_expression.py`：`ecasp prepare-rbp-expression` 子命令的实现与条件向量读写工具。
  - `openspliceai/variant/variant.py`：VCF 注释入口。
- 若遇到问题或希望贡献功能，欢迎在 GitHub Issues 中提问。

祝你在 ECASP 相关研究中取得好结果。
