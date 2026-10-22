#!/bin/bash
# Stage B, one run per pod: pull SFT model -> train 200 steps -> fseval (n=256,
# deployed scale) + expr-only probe (k=1) [+ base-model evals on s9 / s16] -> push.
#   bash scripts/pod/run_rl.sh <run_name>
set -euo pipefail
run=${1:?run_name}
source "$(dirname "$0")/env.sh"
cd "$RL"; source .venv/bin/activate
[ -f "$GRAFT_COUNTDOWN_SFT_MODEL/config.json" ] || bash "$(dirname "$0")/sync.sh" pull_sft
case $run in
  cdhf100_gr_*) scale=0.0 ;;
  countdown_code_hack_*) scale=1.0 ;;
  *) echo "unknown run $run"; exit 1 ;;
esac
ck=output/repro_fig2/$run/checkpoint-200
if [ ! -d "$ck" ]; then
  REPRO_RUNS=$run python -u sweep.py --name repro_fig2 --config sweeps/repro_countdown_fig2.py \
    --no_baseline --no-eval_at_start --overwrite_run
fi
test -d "$ck"
python tools/local_fseval.py --sweep_dir output/repro_fig2 --only_exact "$run" --n_eval 256 --scales $scale
python tools/expr_only_probe.py --ckpt "$ck" --k 1 --n 256 --scales $scale --out output/expr_only_probe_arms
if [[ $run == cdhf100_gr_lccoh64_lr3_s9 ]]; then   # base model (SFT) hack rate, as in the paper
  python tools/local_fseval.py --sweep_dir output/repro_fig2 --only_exact "$run" --n_eval 256 \
    --retain_scale 0.0 --scales 0.0
fi
if [[ $run == cdhf100_gr_lccoh64_lr3_s16 ]]; then  # base model expr-only pass@1
  python tools/expr_only_probe.py --ckpt "$ck" --k 1 --n 256 --retain_scale 0.0 --scales 0.0 \
    --out output/expr_only_probe_arms
fi
bash "$(dirname "$0")/sync.sh" push
echo "RUN_RL_OK $run"
