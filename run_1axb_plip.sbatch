#!/bin/bash
#SBATCH --job-name=boltz_1axb_plip
#SBATCH --partition=bme.gpumolml.q
#SBATCH --output=logs/boltz_1axb_plip_%j.out
#SBATCH --error=logs/boltz_1axb_plip_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Activate conda environment and set library paths
source /home/20255165/miniforge3/etc/profile.d/conda.sh
conda activate cofolding_dynamics_cuda

# CRITICAL: Set LD_LIBRARY_PATH to use conda environment's libraries BEFORE running Python
# This fixes the libstdc++ GLIBCXX_3.4.29 error
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=""  # Clear any preloaded libraries that might conflict

# Verify the library path is set correctly
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Verify libstdc++ version available
echo "Checking libstdc++ version in conda environment:"
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX | tail -5
echo ""

# Change to working directory
cd /vast.mnt/home/20255165/boltz-hackathon-template

# Create logs directory if it doesn't exist
mkdir -p logs

# Print environment info
echo "=========================================="
echo "Environment Information"
echo "=========================================="

echo "Python version: $(python --version)"
echo "RDKit version: $(python -c 'import rdkit; print(rdkit.__version__)')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else \"N/A\")')"
echo ""
echo "=========================================="
echo ""

# Input and output
INPUT_YAML="examples/1axb_chain_subset.yaml"
OUTPUT_DIR="results_1axb_plip"

echo "Running Boltz prediction with PLIP interaction constraints"
echo "Input: $INPUT_YAML"
echo "Output: $OUTPUT_DIR"
echo ""
echo "This prediction includes:"
echo "  - 7 PLIP-derived interaction constraints"
echo "  - 6 hydrogen bonds"
echo "  - 1 hydrophobic contact"
echo ""

# Run Boltz prediction with PLIP constraints
# The constraints will be automatically detected and used
python -m boltz.main predict "$INPUT_YAML" \
    --out_dir "$OUTPUT_DIR" \
    --model boltz2 \
    --accelerator gpu \
    --devices 1 \
    --diffusion_samples 5 \
    --use_potentials \
    --use_msa_server \
    --output_format mmcif \
    --seed 42 \
    --write_full_pae

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Job completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    
    # List output files
    echo "Output files:"
    find "$OUTPUT_DIR" -type f -name "*.cif" -o -name "*.json" -o -name "*.npz" 2>/dev/null | head -20
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Job failed with exit code: $?"
    echo "Check the error log for details."
    echo "=========================================="
    exit 1
fi

echo ""
echo "End Time: $(date)"
echo ""

