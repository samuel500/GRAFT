#!/bin/bash
# Several exp/* runs of one sweep queued on ONE pod, by run name (run_seed_queue.sh
# from exp/dn-anchor, keyed on EXP_RUNS instead of seeds): pull SFT model ->
# smoke (unit tests + 4 steps of the first run + tiny fseval/probe) -> per run:
# train 200 steps -> fseval (n=256) at forget 0.0 and 1.0 + expr-only probe (k=1)
# at 0.0 -> push to the network volume. A failed run is logged and the queue moves
# on. Writes $LOGS/<sweep_name>.done (EXIT=0|1 + failed runs) at the end, also
# when the smoke fails, so a watcher can stop the pod.
# Results go to $GRAFT_S3_URI (set it, e.g. s3://<volume-id>/graft-exp); the SFT
# model is pulled from graft-repro and kept outside rl/output so it isn't pushed.
#   bash scripts/pod/run_exp_queue.sh <sweep_file> <sweep_name> "<run> [<run> ...]"
set -euo pipefail
sweep_file=${1:?sweep_file}; name=${2:?sweep_name}; runs=${3:?runs}
: "${GRAFT_S3_URI:?set GRAFT_S3_URI}"
source "$(dirname "$0")/env.sh"
cd "$RL"; source .venv/bin/activate
done_f="$LOGS/$name.done"
trap 'echo "EXIT=1 aborted at line $LINENO" > "$done_f"' ERR
[ -f "$GRAFT_COUNTDOWN_SFT_MODEL/config.json" ] ||
  GRAFT_S3_URI=s3://<volume-id>/graft-repro bash "$(dirname "$0")/s3_sync.sh" pull_sft
SW="--config $sweep_file --no_baseline --no-eval_at_start --overwrite_run"

python -m pytest -q -p no:cacheprovider tests/test_loreft_adapter.py tests/test_split_moment_capture.py \
  tests/test_fused_routing_equivalence.py tests/test_routing_masks.py tests/test_split_moment_optim.py
smoke=${runs%% *}
EXP_RUNS=$smoke python -u sweep.py --name smoke_$name $SW --no_wandb --max_steps 4 --save_steps 4
test -d output/smoke_$name/$smoke/checkpoint-4
python tools/local_fseval.py --sweep_dir output/smoke_$name --only_exact $smoke \
  --n_eval 16 --scales 0.0,1.0 --overwrite
python tools/expr_only_probe.py --ckpt output/smoke_$name/$smoke/checkpoint-4 --k 1 --n 16 \
  --scales 0.0 --out output/smoke_${name}_probe
rm -rf output/smoke_$name output/smoke_${name}_fseval output/smoke_${name}_probe
echo SMOKE_OK
trap - ERR

run_one() {  # && chain: set -e does not apply inside a tested function call
  local run=$1
  local ck=output/$name/$run/checkpoint-200
  { [ -d "$ck" ] || EXP_RUNS=$run python -u sweep.py --name $name $SW; } &&
    test -d "$ck" &&
    python tools/local_fseval.py --sweep_dir output/$name --only_exact "$run" \
      --n_eval 256 --scales 0.0,1.0 &&
    python tools/expr_only_probe.py --ckpt "$ck" --k 1 --n 256 --scales 0.0 \
      --out output/expr_only_probe_$name
}

failed=""
for run in $runs; do
  echo "RUN_START $run $(date -u +%FT%TZ)"
  run_one $run && echo "RUN_DONE $run $(date -u +%FT%TZ)" || { failed="$failed $run"; echo "RUN_FAIL $run"; }
  bash "$(dirname "$0")/s3_sync.sh" push || echo "PUSH_FAIL after $run"
done
ex=0; [ -n "$failed" ] && ex=1
echo "EXIT=$ex failed:${failed:- none}" > "$done_f"
echo "QUEUE_DONE $name EXIT=$ex failed:${failed:- none}"
