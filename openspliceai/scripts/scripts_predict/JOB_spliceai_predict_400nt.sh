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

for num in 11 12 22 40 42; do
    python predict_spliceaitoolkit.py --flanking-size 400 \
    --project-name spliceai_pytorch_prediction_rs$num \
    --test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE_test/dataset_test.h5 \
    --output-dir /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_predict_outdir/ \
    --model /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/models/spliceai-mane/400nt/model_400nt_rs$num.pt
done

# dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/dataset_test.h5 \

# python predict_spliceai27.py --flanking-size 400 \
# --project-name spliceai5_prediction \
# --test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE_test/dataset_test.h5 \
# --output-dir /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_predict_outdir/ \
# --model /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/models/spliceai/spliceai5.h5 -d

