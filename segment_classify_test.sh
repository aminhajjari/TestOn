#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=MILK10k-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon/logs/%x-%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon/logs/%x-%j.err

# Setup
PROJECT_DIR="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"
SCRIPT_DIR="$PROJECT_DIR/SegCon"
VENV_DIR="$PROJECT_DIR/venv"

cd "$SCRIPT_DIR" || exit 1

# Environment variables
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export DATASET_PATH="$PROJECT_DIR/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="$PROJECT_DIR/MILK10k_Training_GroundTruth.csv"
export OUTPUT_PATH="$PROJECT_DIR/outputs_test"
export SAM2_MODEL_PATH="$PROJECT_DIR/segment-anything-2"
export CONCEPTCLIP_MODEL_PATH="$PROJECT_DIR/ConceptModel"
export HUGGINGFACE_CACHE_PATH="$PROJECT_DIR/huggingface_cache"

# Load modules
module --force purge
module load StdEnv/2023 python/3.11 cuda/11.8 cudnn/8.9.7 opencv/4.12.0

# Activate virtual environment
source "$VENV_DIR/bin/activate" || exit 1

# Install dependencies
pip install --no-index torch torchvision torchaudio transformers pillow pandas numpy opencv-python-headless pydicom nibabel matplotlib seaborn tqdm simpleitk scikit-learn
pip install --no-index -e "$SAM2_MODEL_PATH"

# Pre-execution checks
[ ! -d "$DATASET_PATH" ] && { echo "ERROR: Dataset path missing: $DATASET_PATH"; exit 1; }
[ ! -f "path_colored.py" ] && { echo "ERROR: path_colored.py not found"; exit 1; }
[ ! -d "$SAM2_MODEL_PATH" ] && { echo "ERROR: SAM2 path missing: $SAM2_MODEL_PATH"; exit 1; }
[ ! -f "$SAM2_MODEL_PATH/checkpoints/sam2_hiera_large.pt" ] && { echo "ERROR: Checkpoint missing: $SAM2_MODEL_PATH/checkpoints/sam2_hiera_large.pt"; exit 1; }
[ ! -d "$CONCEPTCLIP_MODEL_PATH" ] && { echo "ERROR: ConceptCLIP path missing: $CONCEPTCLIP_MODEL_PATH"; exit 1; }
[ ! -d "$HUGGINGFACE_CACHE_PATH" ] && mkdir -p "$HUGGINGFACE_CACHE_PATH"
[ ! -f "$GROUNDTRUTH_PATH" ] && echo "WARNING: Ground truth file missing: $GROUNDTRUTH_PATH"
mkdir -p "$OUTPUT_PATH"

# Run pipeline in test mode
srun --gres=gpu:1 python path_colored.py --test 2>&1 | tee "${OUTPUT_PATH}/pipeline_log_colored.txt"
EXIT_CODE=${PIPESTATUS[0]}

# Post-execution
if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline completed successfully!"
    [ -f "${OUTPUT_PATH}/reports/detailed_results.csv" ] && echo "Processed images: $(( $(wc -l < "${OUTPUT_PATH}/reports/detailed_results.csv") - 1 ))"
    [ -f "${OUTPUT_PATH}/reports/processing_report.json" ] && echo "Processing report generated"
    [ -d "${OUTPUT_PATH}/segmented_for_conceptclip" ] && echo "Segmented outputs: $(find "${OUTPUT_PATH}/segmented_for_conceptclip" -name "*.png" | wc -l) files"
    [ -f "${OUTPUT_PATH}/visualizations/summary_plots.png" ] && echo "Summary plots generated"
else
    echo "Pipeline failed with exit code: $EXIT_CODE"
fi

echo "Output directory contents:"
ls -la "$OUTPUT_PATH/"
echo "Job End Time: $(date)"
exit $EXIT_CODE