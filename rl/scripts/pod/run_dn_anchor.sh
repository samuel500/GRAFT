#!/bin/bash
# Anchored no-intervention arm (sweeps/countdown_hf100_dn_anchor.py), all 4 seeds
# queued on ONE pod: pull SFT model -> 4-step smoke of the forget-less config ->
# per seed: train 200 steps -> fseval (n=256, scale 1.0) + expr-only probe (k=1)
# -> push to the network volume. A failed seed is logged and the queue moves on.
# Writes $LOGS/dn_anchor.done (EXIT=0|1 + failed seeds) at the end, also when the
# smoke fails, so a watcher can stop the pod.
#   bash scripts/pod/run_dn_anchor.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$RL"; source .venv/bin/activate
trap 'echo "EXIT=1 aborted at line $LINENO" > "$LOGS/dn_anchor.done"' ERR
[ -f "$GRAFT_COUNTDOWN_SFT_MODEL/config.json" ] || bash "$(dirname "$0")/s3_sync.sh" pull_sft
SW="--config sweeps/countdown_hf100_dn_anchor.py --no_baseline --no-eval_at_start --overwrite_run"

# Smoke: m128_retain_only + classic routing at recall 0 has not run before.
smoke=cdhf100_dn_lccoh64_lr3_s9
REPRO_RUNS=$smoke python -u sweep.py --name smoke_dn_anchor $SW --no_wandb --max_steps 4 --save_steps 4
test -d output/smoke_dn_anchor/$smoke/checkpoint-4
python tools/local_fseval.py --sweep_dir output/smoke_dn_anchor --only_exact $smoke \
  --n_eval 16 --scales 1.0 --overwrite
rm -rf output/smoke_dn_anchor output/smoke_dn_anchor_fseval
echo SMOKE_OK
trap - ERR

run_seed() {  # && chain: set -e does not apply inside a tested function call
  local run=cdhf100_dn_lccoh64_lr3_s$1
  local ck=output/dn_anchor/$run/checkpoint-200
  { [ -d "$ck" ] || REPRO_RUNS=$run python -u sweep.py --name dn_anchor $SW; } &&
    test -d "$ck" &&
    python tools/local_fseval.py --sweep_dir output/dn_anchor --only_exact "$run" \
      --n_eval 256 --scales 1.0 &&
    python tools/expr_only_probe.py --ckpt "$ck" --k 1 --n 256 --scales 1.0 \
      --out output/expr_only_probe_arms
}

failed=""
for s in 9 15 16 1; do
  run_seed $s || { failed="$failed $s"; echo "SEED_FAIL $s"; }
  bash "$(dirname "$0")/s3_sync.sh" push || echo "PUSH_FAIL after seed $s"
done
ex=0; [ -n "$failed" ] && ex=1
echo "EXIT=$ex failed:${failed:- none}" > "$LOGS/dn_anchor.done"
echo "DN_ANCHOR_DONE EXIT=$ex failed:${failed:- none}"
