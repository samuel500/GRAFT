#!/bin/bash
# Steering-variance pipeline on one GPU pod: pull the SFT base and a
# no-intervention adapter (Hugging Face, sha256-checked), smoke every phase on a
# few questions, run the full pipeline, push results (scripts/pod/sync.sh; a
# no-op unless GRAFT_SYNC_URI is set). Run after rl/scripts/pod/setup.sh
# (GRAFT_ROOT=/workspace/graft-exp). Writes $LOGS/sv.done (SV_OK / SV_FAIL ...).
#   tmux new -d -s sv "bash experiments/steering_variance/run_pod.sh > /workspace/graft-exp/logs/sv.log 2>&1"
# Another adapter: HF_ADAPTER=no-intervention/<run> OUT=output/steering_variance_<run>
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export GRAFT_ROOT=${GRAFT_ROOT:-/workspace/graft-exp}
# The checkpoint's run_config.yaml names the base model by this path.
export GRAFT_COUNTDOWN_SFT_MODEL=${GRAFT_COUNTDOWN_SFT_MODEL:-/workspace/graft-repro/GRAFT/rl/output/countdown_sft_model/qwen3-8b}
source "$GRAFT_ROOT/GRAFT/rl/scripts/pod/env.sh"
cd "$RL"; source .venv/bin/activate
HF_ADAPTER=${HF_ADAPTER:-no-intervention/cdhf100_dn_lr3_s9}
HF_ADAPTER_REPO=${HF_ADAPTER_REPO:-samuel500/graft-repro-adapters}
HF_ADAPTER_REV=${HF_ADAPTER_REV:-f806ba0f760568b61ebe093097337b0e61bed9b0}
CKPT=${CKPT:-/workspace/hf_adapters/$(basename "$HF_ADAPTER")/checkpoint-200}
OUT=${OUT:-output/steering_variance}
SMOKE_OUT=output/steering_variance_smoke
finish() { echo "$1" | tee "$LOGS/sv.done"; exit "${2:-0}"; }

if [ ! -f "$CKPT/model.safetensors" ]; then
  python - "$HF_ADAPTER_REPO" "$HF_ADAPTER_REV" "$HF_ADAPTER" "$CKPT" <<'PY' || finish "SV_FAIL adapter download" 1
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
fi
if [ ! -f "$GRAFT_COUNTDOWN_SFT_MODEL/config.json" ]; then
  python -c "import sys; from huggingface_hub import snapshot_download as d; \
d(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])" \
    "$GRAFT_SFT_REPO" "$GRAFT_SFT_REV" "$GRAFT_COUNTDOWN_SFT_MODEL" || finish "SV_FAIL SFT download" 1
  echo "$GRAFT_SFT_SHA256  $GRAFT_COUNTDOWN_SFT_MODEL/model.safetensors" | sha256sum -c - \
    || finish "SV_FAIL SFT sha256" 1
fi

sv() { python -u "$here/steering_variance.py" "$@"; }
pipeline() {  # $1 = out dir, $2 = log prefix, rest = extra flags
  local out=$1 tag=$2; shift 2
  for step in "pps" "rollouts --source base" "rollouts --source dn --ckpt $CKPT" \
              "capture --model base" "capture --model dn --ckpt $CKPT" "analyze"; do
    echo "SV_STEP $tag $step $(date -u +%FT%TZ)"
    sv $step --out "$out" "$@" >> "$LOGS/sv_$tag.log" 2>&1 || return 1
  done
}

# SV_MODE=prompts: only the prompt-token replay (phase prompts) on top of an
# earlier full run; that run's outputs come from the sync mirror if missing here.
if [ "${SV_MODE:-full}" = prompts ]; then
  if [ ! -f "$OUT/rollouts_dn.json" ]; then
    command -v aws >/dev/null || uv tool install awscli
    aws s3 sync ${GRAFT_SYNC_ARGS:-} "${GRAFT_SYNC_URI:?no earlier outputs and no GRAFT_SYNC_URI}/$OUT/" "$OUT/" \
      --exclude "*acts_L20.npy" --exclude "*/proj.npz" --only-show-errors || finish "SV_FAIL pull earlier outputs" 1
  fi
  for step in "prompts --model base" "prompts --model dn --ckpt $CKPT"; do
    echo "SV_STEP prompts $step $(date -u +%FT%TZ)"
    sv $step --out "$OUT" >> "$LOGS/sv_prompts.log" 2>&1 || finish "SV_FAIL prompts (see $LOGS/sv_prompts.log)" 1
  done
  bash scripts/pod/sync.sh push || finish "SV_FAIL push" 1
  finish "SV_OK prompts $HF_ADAPTER $(date -u +%FT%TZ)"
fi

echo "SV_SMOKE $(date -u +%FT%TZ)"
pipeline "$SMOKE_OUT" smoke --smoke || finish "SV_FAIL smoke (see $LOGS/sv_smoke.log)" 1
rm -rf "$SMOKE_OUT"
echo "SV_SMOKE_OK $(date -u +%FT%TZ)"

echo "SV_START $HF_ADAPTER -> $OUT $(date -u +%FT%TZ)"
pipeline "$OUT" full || { bash scripts/pod/sync.sh push; finish "SV_FAIL full (see $LOGS/sv_full.log)" 1; }
bash scripts/pod/sync.sh push || finish "SV_FAIL push" 1
finish "SV_OK $HF_ADAPTER $(date -u +%FT%TZ)"
