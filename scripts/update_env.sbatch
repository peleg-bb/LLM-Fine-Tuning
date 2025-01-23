#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-1:00:00
#SBATCH --job-name=update_env
#SBATCH --output=output/update_env-%J.out
#SBATCH --gpus=rtx_3090:1

module load anaconda
source activate my_env

pip install bert-score rouge-score evaluate nltk
pip install torch
pip install --upgrade transformers peft datasets
pip install evaluate datasets
pip install bitsandbytes