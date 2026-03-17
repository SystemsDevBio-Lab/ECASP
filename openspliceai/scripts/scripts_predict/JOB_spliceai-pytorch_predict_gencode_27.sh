#!/bin/bash
#SBATCH --job-name=JOB_spliceai_pytorch_train_MANE_small_dataset
#SBATCH --time=48:0:0
#SBATCH -p a100 # specify the GPU partition
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=48
#SBATCH --gres=gpu:4
#SBATCH -A ssalzbe1_gpu ### Slurm-account is usually the PI’s userid
#SBATCH --export=ALL


# Load system modules
ml purge
# module load gcc/9.3.0
# module load cuda/11.1.0
# module load anaconda

source activate /home/kchao10/miniconda3/envs/pytorch_cuda

# Check if the Python interpreter picks up packages from both the system and the Conda environment
which python
python -c "import sys; print(sys.path)"

# FLANKING_SIZE=10000

for FLANKING_SIZE in 80 400 2000 10000; do
    SPECIES=SpliceAI27
<<<<<<< HEAD
    RANDOPM_SEED=2
=======
    RANDOPM_SEED=1
>>>>>>> 5d720e67b7b2cf0cdb6e255971dbc97cfd9ae606
    LOSS_FUNC=cross_entropy_loss
    EXP_NUM=full_dataset

    python predict_spliceaitoolkit.py --flanking-size ${FLANKING_SIZE} \
<<<<<<< HEAD
    --project-name spliceai_${SPECIES}_gencode_rs${RANDOPM_SEED} \
    --test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${SPECIES}/dataset_test.h5 \
    --type test \
    --output-dir /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_predict_outdir/ \
    --model /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_train_outdir/SpliceAI_MANE_cross_entropy_loss_w0_${FLANKING_SIZE}_full_dataset_rs${RANDOPM_SEED}/full_dataset/models/best_model.pt
=======
    --project-name spliceai_${SPECIES}_rs${RANDOPM_SEED} \
    --test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${SPECIES}/dataset_test.h5 \
    --type test \
    --output-dir /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_predict_outdir/ \
    --model /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/models/spliceai-mane/${FLANKING_SIZE}nt/model_${FLANKING_SIZE}nt_rs${RANDOPM_SEED}.pt
>>>>>>> 5d720e67b7b2cf0cdb6e255971dbc97cfd9ae606
done

