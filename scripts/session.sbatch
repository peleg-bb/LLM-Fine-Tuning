#!/bin/bash
#SBATCH --partition main                         
#SBATCH --time 0-2:00:00                      
#SBATCH --job-name Mamba_QA                 
#SBATCH --output=output/mamba_session-%J.out        
#SBATCH --mail-user=yuvalx123x@gmail.com      
#SBATCH --mail-type=ALL                       
#SBATCH --gpus=rtx_3090:1                     
#SBATCH --mem=32G                             

echo -e "Job ID             $SLURM_JOBID"
echo -e "Job Nodes         $SLURM_JOB_NODELIST\n"

# Load required modules and initialize conda
module load anaconda
eval "$(conda shell.bash hook)"
conda activate my_env

# Get the project root directory (parent of scripts directory)
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Print directories for debugging
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# Define input and output paths
QUESTIONS_FILE="$PROJECT_ROOT/data/questions.txt"
ANSWERS_FILE="$PROJECT_ROOT/data/answers.txt"

# Verify input file exists
if [ ! -f "$QUESTIONS_FILE" ]; then
    echo "Error: Questions file not found at $QUESTIONS_FILE"
    exit 1
fi

echo "Using questions from: $QUESTIONS_FILE"
echo "Writing answers to: $ANSWERS_FILE"

# Run the session script
python -u "$PROJECT_ROOT/finetuning/session.py" \
    --base-model tiiuae/falcon-mamba-7b \
    --adapter-path "$PROJECT_ROOT/models/mamba-model" \
    --tokenizer-path "$PROJECT_ROOT/models/mamba-tokenizer" \
    --input-file "$QUESTIONS_FILE" \
    --output-file "$ANSWERS_FILE"

echo "Job completed at $(date)"