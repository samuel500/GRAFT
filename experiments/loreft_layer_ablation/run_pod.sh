#!/bin/bash
# LoReFT layer-window ablation battery on one (multi-GPU) pod: pull the
# checkpoint + SFT model, smoke 2 configs at n=8 on GPU 0, then one ablate.py
# shard per GPU, then push results to the network volume. Run after
# rl/scripts/pod/setup.sh (GRAFT_ROOT=/workspace/graft-exp). Writes
# $LOGS/ablate.done (ABLATE_OK / ABLATE_FAIL ...) when finished. The S3 push
# is a sync, so pods sharing the prefix only add their own configs.
#   tmux new -d -s ablate "bash experiments/loreft_layer_ablation/run_pod.sh cdhf100_gr_loreft_r1_lr3_s9 > /workspace/graft-exp/logs/ablate.log 2>&1"
set -uo pipefail
run=${1:?run_name, e.g. cdhf100_gr_loreft_r1_lr3_s9}
here=$(cd "$(dirname "$0")" && pwd)
export GRAFT_ROOT=${GRAFT_ROOT:-/workspace/graft-exp}
export GRAFT_S3_URI=${GRAFT_S3_URI:-s3://<volume-id>/graft-exp}
# The checkpoint's run_config.yaml names the base model by this path.
export GRAFT_COUNTDOWN_SFT_MODEL=${GRAFT_COUNTDOWN_SFT_MODEL:-/workspace/graft-repro/GRAFT/rl/output/countdown_sft_model/qwen3-8b}
source "$GRAFT_ROOT/GRAFT/rl/scripts/pod/env.sh"
cd "$RL"; source .venv/bin/activate
finish() { echo "$1" | tee "$LOGS/ablate.done"; exit "${2:-0}"; }

ck=output/exp_loreft/$run/checkpoint-200
[ -f "$ck/model.safetensors" ] || aws s3 sync $S3_ARGS "$S3_URI/output/exp_loreft/$run/" \
  "output/exp_loreft/$run/" --exclude "*optimizer.pt" --only-show-errors
[ -f "$GRAFT_COUNTDOWN_SFT_MODEL/config.json" ] || aws s3 sync $S3_ARGS \
  s3://<volume-id>/graft-repro/output/countdown_sft_model/qwen3-8b/ "$GRAFT_COUNTDOWN_SFT_MODEL/" --only-show-errors
test -f "$ck/model.safetensors" || finish "ABLATE_FAIL no checkpoint $ck" 1
test -f "$GRAFT_COUNTDOWN_SFT_MODEL/config.json" || finish "ABLATE_FAIL no SFT model" 1

echo "ABLATE_SMOKE $(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES=0 python -u "$here/ablate.py" --ckpt "$ck" --n 8 \
  --out output/loreft_layer_ablation_smoke --only retain_keep_1-5,forget_drop_31-35 \
  > "$LOGS/ablate_smoke.log" 2>&1 || finish "ABLATE_FAIL smoke (see $LOGS/ablate_smoke.log)" 1
echo "ABLATE_SMOKE_OK $(date -u +%FT%TZ)"

# One shard per local GPU. Across several pods, SHARD_OFFSET/TOTAL_SHARDS give
# this pod's GPUs the global shards offset..offset+n-1 of TOTAL_SHARDS.
n=$(nvidia-smi -L | wc -l)
off=${SHARD_OFFSET:-0}; total=${TOTAL_SHARDS:-$n}
echo "ABLATE_START $run shards=$off..$((off + n - 1))/$total $(date -u +%FT%TZ)"
pids=()
for i in $(seq 0 $((n - 1))); do
  s=$((off + i))
  CUDA_VISIBLE_DEVICES=$i python -u "$here/ablate.py" --ckpt "$ck" \
    --out output/loreft_layer_ablation --shard "$s" --num_shards "$total" \
    > "$LOGS/ablate_${run}_shard$s.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
bash scripts/pod/s3_sync.sh push
done_n=$(ls output/loreft_layer_ablation/$run/configs/*.json 2>/dev/null | wc -l)
[ $fail = 0 ] && finish "ABLATE_OK $run configs=$done_n $(date -u +%FT%TZ)" \
  || finish "ABLATE_FAIL $run configs=$done_n (see $LOGS/ablate_${run}_shard*.log)" 1
