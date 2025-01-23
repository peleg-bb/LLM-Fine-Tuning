#!/bin/bash
#SBATCH --partition main                         
#SBATCH --time 0-4:00:00                      
#SBATCH --job-name Mamba_Eval                 
#SBATCH --output=output/mamba_eval-%J.out        
#SBATCH --mail-user=kopansev@post.bgu.ac.il     
#SBATCH --mail-type=ALL                       
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=32G                             

echo -e "Job ID             $SLURM_JOBID"
echo -e "Job Nodes         $SLURM_JOB_NODELIST\n"

module load anaconda
source activate my_env

python -u finetuning/eval.py