#!/bin/bash
#SBATCH --partition=main                         
#SBATCH --time=0-2:00:00                      
#SBATCH --job-name=setup_reset
#SBATCH --output=output/setup_reset.out                                  
#SBATCH --gpus=rtx_3090:1                            

ENV_NAME="my_env"

module load anaconda

echo -e "Job ID         $SLURM_JOBID"
echo -e "Job Node    $SLURM_JOB_NODELIST"

echo "Activating environment .. "
conda activate ${ENV_NAME}

echo "Cleaning environment..."
# List currently installed pip packages
pip freeze > current_pip_packages.txt

echo "Uninstalling packages not listed in requirements.txt ..."
grep -vxFf requirements.txt current_pip_packages.txt | xargs -r pip uninstall -y

echo "Reinstalling packages from requirements.txt..."
pip install -r requirements.txt

echo "Environment is ready !!!"


