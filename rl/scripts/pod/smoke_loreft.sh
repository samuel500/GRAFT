#!/bin/bash
# LoReFT composite smoke (exp checkout at /workspace/graft-exp): unit tests,
# 8-step r8 GRAFT run (2 rollouts) from the SFT model with W&B off, then the
# posthoc loaders. Note: the vLLM-vs-HF drift metrics (sampling/*) only go to
# W&B, so this smoke does not measure them.
set -euo pipefail
source /workspace/graft-repro/GRAFT/rl/.venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
export RH_REPO_PATH=/workspace/graft-repro/ext/rl-rewardhacking CC_REPO=/workspace/graft-repro/ext/Countdown-Code
export GRAFT_COUNTDOWN_SFT_MODEL=/workspace/graft-repro/GRAFT/rl/output/countdown_sft_model/qwen3-8b
cd /workspace/graft-exp/GRAFT/rl
python -m pytest -q -p no:cacheprovider tests/test_loreft_adapter.py tests/test_split_moment_capture.py tests/test_fused_routing_equivalence.py tests/test_routing_masks.py tests/test_split_moment_optim.py
run=cdhf100_gr_loreft_r8_lr1_s9
EXP_RUNS=$run python -u sweep.py --name smoke_loreft --config sweeps/countdown_hf100_gr_loreft.py \
  --no_baseline --no-eval_at_start --no_wandb --overwrite_run --max_steps 8 --save_steps 8
test -d output/smoke_loreft/$run/checkpoint-8
python tools/local_fseval.py --sweep_dir output/smoke_loreft --only_exact $run --n_eval 16 --scales 0.0,1.0 --overwrite
python tools/expr_only_probe.py --ckpt output/smoke_loreft/$run/checkpoint-8 --k 1 --n 16 --scales 0.0 --out output/smoke_loreft_probe
echo SMOKE_LOREFT_OK
