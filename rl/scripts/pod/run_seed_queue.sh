#!/bin/bash
# All seeds of one Countdown sweep queued on ONE pod (run_dn_anchor.sh, generalized):
# pull SFT model -> 4-step smoke -> per seed: train 200 steps -> fseval (n=256) +
# expr-only probe (k=1) at <scale> -> optional results push (sync.sh). A failed seed
# is logged and the queue moves on. Writes $LOGS/<sweep_name>.done (EXIT=0|1 +
# failed seeds) at the end, also when the smoke fails, so a watcher can stop the pod.
#   bash scripts/pod/run_seed_queue.sh <sweep_file> <sweep_name> <run_prefix> <scale> "<seeds>"
#   e.g. run_seed_queue.sh sweeps/countdown_hf100_dn_lr3.py dn_lr3 cdhf100_dn_lr3 1.0 "9 15 16 1"
set -euo pipefail
sweep_file=${1:?sweep_file}; name=${2:?sweep_name}; prefix=${3:?run_prefix}
scale=${4:?scale}; seeds=${5:?seeds}
source "$(dirname "$0")/env.sh"
cd "$RL"; source .venv/bin/activate
done_f="$LOGS/$name.done"
trap 'echo "EXIT=1 aborted at line $LINENO" > "$done_f"' ERR
[ -f "$GRAFT_COUNTDOWN_SFT_MODEL/config.json" ] || bash "$(dirname "$0")/sync.sh" pull_sft
SW="--config $sweep_file --no_baseline --no-eval_at_start --overwrite_run"

smoke=${prefix}_s${seeds%% *}
REPRO_RUNS=$smoke python -u sweep.py --name smoke_$name $SW --no_wandb --max_steps 4 --save_steps 4
test -d output/smoke_$name/$smoke/checkpoint-4
python tools/local_fseval.py --sweep_dir output/smoke_$name --only_exact $smoke \
  --n_eval 16 --scales $scale --overwrite
rm -rf output/smoke_$name output/smoke_${name}_fseval
echo SMOKE_OK
trap - ERR

run_seed() {  # && chain: set -e does not apply inside a tested function call
  local run=${prefix}_s$1
  local ck=output/$name/$run/checkpoint-200
  { [ -d "$ck" ] || REPRO_RUNS=$run python -u sweep.py --name $name $SW; } &&
    test -d "$ck" &&
    python tools/local_fseval.py --sweep_dir output/$name --only_exact "$run" \
      --n_eval 256 --scales $scale &&
    python tools/expr_only_probe.py --ckpt "$ck" --k 1 --n 256 --scales $scale \
      --out output/expr_only_probe_arms
}

failed=""
for s in $seeds; do
  run_seed $s || { failed="$failed $s"; echo "SEED_FAIL $s"; }
  bash "$(dirname "$0")/sync.sh" push || echo "PUSH_FAIL after seed $s"
done
ex=0; [ -n "$failed" ] && ex=1
echo "EXIT=$ex failed:${failed:- none}" > "$done_f"
echo "QUEUE_DONE $name EXIT=$ex failed:${failed:- none}"
