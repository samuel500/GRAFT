#!/bin/bash
# Idempotent pod setup: uv venv (+ vLLM patches), rl-rewardhacking, awscli.
#   bash scripts/pod/setup.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$ROOT/ext" "$HF_HOME"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd "$RL"
uv sync
source .venv/bin/activate
bash vllm_patches/apply.sh
[ -d "$RH_REPO_PATH" ] || git clone --depth 1 https://github.com/ariahw/rl-rewardhacking "$RH_REPO_PATH"
test -f "$RH_REPO_PATH/results/data/leetcode_train_medhard_filtered.jsonl"
command -v aws >/dev/null || uv tool install awscli
python -c "import torch, vllm, flash_attn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'vllm', vllm.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "nproc=$(nproc) mem=$(free -g | awk '/Mem/{print $2}')G"; df -h /workspace | tail -1
echo SETUP_OK
