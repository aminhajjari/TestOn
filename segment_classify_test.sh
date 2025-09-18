#!/bin/bash
#SBATCH --job-name=download-medgemma
#SBATCH --partition=accel
#SBATCH --gres=gpu:1              # not really needed, but ok
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=download.out

module load python/3.10
pip install --no-cache-dir huggingface_hub safetensors

# Download and save MedGemma 4B IT multimodal
python << 'PYCODE'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="google/medgemma-4b-it",
    cache_dir="/scratch/$USER/medgemma-4b-it",  # permanent path
    local_dir="/scratch/$USER/medgemma-4b-it",  # explicitly save here
    local_dir_use_symlinks=False
)
print("✅ MedGemma-4B saved under /scratch/$USER/medgemma-4b-it")
PYCODE
