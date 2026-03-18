#!/bin/bash
#SBATCH --job-name=JOB_spliceai_pytorch_create_MANE_dataset
#SBATCH --partition=parallel
#SBATCH -t 02-01:30:15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --account=ssalzbe1
#SBATCH --cpus-per-task=6
#SBATCH --export=ALL


# Load system modules
ml purge
module load gcc/9.3.0
# module load python/3.11.6
# module load py-h5py/3.8.0
# module load cuda/11.1.0
# module load pyTorch/1.8.1-cuda-11.1.1
# module load anaconda
ml

conda activate /home/kchao10/miniconda3/envs/spliceai

# Check if the Python interpreter picks up packages from both the system and the Conda environment
which python
python -c "import sys; print(sys.path)"

python Step_2_create_dataset_noh5py.py \
--train-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/datafile_train.h5 \
--test-dataset  /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/datafile_test.h5 \
--output-dir /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/ \
> dataset_creation.log 2> dataset_creation_error.log