#!/bin/bash
#SBATCH --partition main                         
#SBATCH --time 1-00:00:00                      # Requesting 24 hours as it's a larger model
#SBATCH --job-name Mamba_SFT            
#SBATCH --output=output/mamba_training-%J.out              
#SBATCH --mail-user=yuvalx123x@gmail.com        
#SBATCH --mail-type=ALL                       
#SBATCH --gpus=rtx_3090:1                            
#SBATCH --mem=32G                               # Requesting more memory for the larger model

echo -e "Job ID             $SLURM_JOBID"
echo -e "Job Nodes         $SLURM_JOB_NODELIST\n"

# Load required modules
module load anaconda

# Activate the conda environment
source activate my_env

# Create necessary directories if they don't exist
mkdir -p models/falcon
mkdir -p logs
mkdir -p output

# Print some debug information
echo "Starting training at $(date)"
echo "Python path: $(which python)"
echo "Current directory: $(pwd)"

# Run the training script
python finetuning/train_mamba.py

echo "Training completed at $(date)"