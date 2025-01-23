#!/bin/bash
#SBATCH --partition main                         
#SBATCH --time 0-12:00:00                      
#SBATCH --job-name save_version            
#SBATCH --output=output/%J-save-version.out                               
#SBATCH --gpus=rtx_3090:1                            

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_ARCHIVE="backup/valid_version.tar.gz/valid_version_${TIMESTAMP}.tar.gz"

mkdir -p "backup"
mkdir -p "backup/valid_version.tar.gz"

echo "Saving the current version as a timestamped backup..."
tar -czf "${BACKUP_ARCHIVE}" -C "." .
echo "Backup completed"
