#!/bin/bash
# Pipeline smoke on the plain Qwen3-8B (1 rollout = 4 optimizer steps per arm),
# then the two eval tools on the smoke checkpoints with tiny n.
#   bash scripts/pod/smoke.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$RL"; source .venv/bin/activate
python -m pytest -q tests/test_split_moment_optim.py tests/test_routing_masks.py tests/test_routed_advantages.py
for run in cdhf100_gr_lccoh64_lr3_s9 countdown_code_hack_reward_penalty_amountmissing_s9; do
  GRAFT_COUNTDOWN_SFT_MODEL=Qwen/Qwen3-8B REPRO_RUNS=$run python -u sweep.py --name smoke \
    --config sweeps/repro_countdown_fig2.py --no_baseline --no-eval_at_start --no_wandb \
    --overwrite_run --max_steps 4 --save_steps 4
  test -d output/smoke/$run/checkpoint-4
done
python tools/local_fseval.py --sweep_dir output/smoke --only_exact cdhf100_gr_lccoh64_lr3_s9 \
  --n_eval 16 --scales 0.0 --overwrite
python tools/expr_only_probe.py --ckpt output/smoke/cdhf100_gr_lccoh64_lr3_s9/checkpoint-4 \
  --k 1 --n 16 --scales 0.0 --out output/smoke_probe
echo SMOKE_OK
