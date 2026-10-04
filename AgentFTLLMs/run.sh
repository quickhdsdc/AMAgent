#!/bin/bash
#
#SBATCH --job-name=AMT
#SBATCH --output=AMT_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=36:00:00

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ── Python environment ───────────────────────────────────────────────────────
# uv-managed venv; created once with `uv sync` on a login node.
VENV_BIN="$PROJECT_DIR/.venv/bin"
if [[ ! -f "$VENV_BIN/python" ]]; then
    echo "ERROR: .venv not found. Run 'uv sync' on a login node first." >&2
    exit 1
fi
source "$VENV_BIN/activate"

# ── CUDA ─────────────────────────────────────────────────────────────────────
# cuda/12.4 matches the PyTorch CUDA 12.4 wheels in pyproject.toml.
# Adjust if the cluster uses a different module name.
module load cuda/12.4

# ── env variables ────────────────────────────────────────────────────────────
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_DIR/.cache/triton}"
mkdir -p "$TRITON_CACHE_DIR"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
# Supply HFTOKEN or HUGGINGFACE_HUB_TOKEN through the job environment when needed.
export HFTOKEN="${HFTOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Make `from tasks.AM_defect_classification...` resolve when running a
# script that lives under scripts/.
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ── Experiment ───────────────────────────────────────────────────────────────
# EXPERIMENT is the basename (no .py) of a script under scripts/.
# Examples:
#   sbatch --export=ALL,EXPERIMENT=run_C5C7C10C11_sc,AM_METHOD=C10 run.sh
#   sbatch --export=ALL,EXPERIMENT=run_C4C6_clm_binary,AM_METHOD=C4 \
#          --gres=gpu:a100:1 run.sh
#   sbatch --export=ALL,EXPERIMENT=run_C8_zeroshot_qwen3 \
#          --gres=gpu:a180:1 --output=logs/AMT_log_C8.txt run.sh
EXPERIMENT="${EXPERIMENT:-run_C5C7C10C11_sc}"
python "$PROJECT_DIR/scripts/${EXPERIMENT}.py"
