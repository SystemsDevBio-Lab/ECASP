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
    SPECIES=mouse
    RANDOPM_SEED=12
    LOSS_FUNC=cross_entropy_loss
    EXP_NUM=full_dataset
    for num in 1 2 3 4 5; do
        python predict_spliceai27.py --flanking-size ${FLANKING_SIZE} \
        --project-name spliceai${num}_${SPECIES} \
        --test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${SPECIES}/dataset_test.h5 \
        --output-dir /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_predict_outdir/ \
        --model /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/models/spliceai/${FLANKING_SIZE}nt/spliceai$num.h5
    done
done

