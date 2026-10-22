#!/bin/bash
# Layer-window ablation battery on one (multi-GPU) pod: pull the checkpoint
# (volume S3, or Hugging Face with HF_ADAPTER) + the SFT model (Hugging Face),
# smoke 2 configs at n=8 on GPU 0, then one ablate.py shard per GPU, then push
# results (scripts/pod/sync.sh; a no-op unless GRAFT_SYNC_URI is set). Run after
# rl/scripts/pod/setup.sh (GRAFT_ROOT=/workspace/graft-exp). Writes
# $LOGS/ablate.done (ABLATE_OK / ABLATE_FAIL ...) when finished. The push is a
# sync, so pods sharing the prefix only add their own configs.
#   tmux new -d -s ablate "bash experiments/loreft_layer_ablation/run_pod.sh cdhf100_gr_loreft_r1_lr3_s9 > /workspace/graft-exp/logs/ablate.log 2>&1"
# An adapter from samuel500/graft-repro-adapters (sha256 checked against its meta.json):
#   CKPT=/workspace/hf_adapters/cdhf100_gr_lccoh64_lr3_s1/checkpoint-200 \
#   HF_ADAPTER=graft/cdhf100_gr_lccoh64_lr3_s1 bash .../run_pod.sh cdhf100_gr_lccoh64_lr3_s1
set -uo pipefail
run=${1:?run_name, e.g. cdhf100_gr_loreft_r1_lr3_s9}
here=$(cd "$(dirname "$0")" && pwd)
export GRAFT_ROOT=${GRAFT_ROOT:-/workspace/graft-exp}
# The checkpoint's run_config.yaml names the base model by this path.
export GRAFT_COUNTDOWN_SFT_MODEL=${GRAFT_COUNTDOWN_SFT_MODEL:-/workspace/graft-repro/GRAFT/rl/output/countdown_sft_model/qwen3-8b}
source "$GRAFT_ROOT/GRAFT/rl/scripts/pod/env.sh"
cd "$RL"; source .venv/bin/activate
# SFT base on Hugging Face (same as the repro branch's env.sh).
GRAFT_SFT_REPO=${GRAFT_SFT_REPO:-samuel500/graft-qwen3-8b-countdown-sft}
GRAFT_SFT_REV=${GRAFT_SFT_REV:-a2d54fdf0aa9fa6ee8e487777ee7d63adaf8ec48}
GRAFT_SFT_SHA256=${GRAFT_SFT_SHA256:-533c4f2b03dce7f21ae655a522ceb30d6781aa158fe4c486565491d3a0bd2904}
HF_ADAPTER_REPO=${HF_ADAPTER_REPO:-samuel500/graft-repro-adapters}
HF_ADAPTER_REV=${HF_ADAPTER_REV:-f806ba0f760568b61ebe093097337b0e61bed9b0}
finish() { echo "$1" | tee "$LOGS/ablate.done"; exit "${2:-0}"; }

ck=${CKPT:-output/exp_loreft/$run/checkpoint-200}
if [ -f "$ck/model.safetensors" ]; then :
elif [ -n "${HF_ADAPTER:-}" ]; then
  python - "$HF_ADAPTER_REPO" "$HF_ADAPTER_REV" "$HF_ADAPTER" "$ck" <<'PY' || finish "ABLATE_FAIL adapter download" 1
import hashlib, json, os, shutil, sys
from huggingface_hub import hf_hub_download
repo, rev, sub, ck = sys.argv[1:]
os.makedirs(ck, exist_ok=True)
for f in ("model.safetensors", "dual_lora_config.json", "run_config.yaml", "meta.json"):
    shutil.copy(hf_hub_download(repo, f"{sub}/{f}", revision=rev), os.path.join(ck, f))
want = json.load(open(os.path.join(ck, "meta.json")))["model_safetensors_sha256"]
h = hashlib.sha256()
with open(os.path.join(ck, "model.safetensors"), "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
assert h.hexdigest() == want, f"adapter sha256 {h.hexdigest()} != {want}"
print(f"adapter {repo}@{rev}:{sub} -> {ck}, sha256 ok")
PY
else
  aws s3 sync ${GRAFT_SYNC_ARGS:-} "${GRAFT_SYNC_URI:?no checkpoint: set CKPT, HF_ADAPTER or GRAFT_SYNC_URI}/output/exp_loreft/$run/" \
    "output/exp_loreft/$run/" --exclude "*optimizer.pt" --only-show-errors
fi
if [ ! -f "$GRAFT_COUNTDOWN_SFT_MODEL/config.json" ]; then
  python -c "import sys; from huggingface_hub import snapshot_download as d; \
d(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])" \
    "$GRAFT_SFT_REPO" "$GRAFT_SFT_REV" "$GRAFT_COUNTDOWN_SFT_MODEL" || finish "ABLATE_FAIL SFT download" 1
  echo "$GRAFT_SFT_SHA256  $GRAFT_COUNTDOWN_SFT_MODEL/model.safetensors" | sha256sum -c - \
    || finish "ABLATE_FAIL SFT sha256" 1
fi
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
bash scripts/pod/sync.sh push
done_n=$(ls output/loreft_layer_ablation/$run/configs/*.json 2>/dev/null | wc -l)
[ $fail = 0 ] && finish "ABLATE_OK $run configs=$done_n $(date -u +%FT%TZ)" \
  || finish "ABLATE_FAIL $run configs=$done_n (see $LOGS/ablate_${run}_shard*.log)" 1
