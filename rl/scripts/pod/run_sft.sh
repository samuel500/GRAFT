#!/bin/bash
# Stage A after setup + smoke: SFT data -> 20-step benchmark -> full SFT (+merge)
# -> hack base-rate gate -> push to the network volume.
#   bash scripts/pod/run_sft.sh [data|bench|train|gate|push|all]
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$RL"; source .venv/bin/activate
step=${1:-all}
if [[ $step == data || $step == all ]]; then
  [ -d "$CC_REPO" ] || git clone --depth 1 https://github.com/zohaib-khan5040/Countdown-Code "$CC_REPO"
  python tools/countdown_build_sft_data.py --src "$CC_REPO/o4-mini-distillation-16k.jsonl" \
    --out_dir output/countdown_sft
  wc -l output/countdown_sft/{train,val}.jsonl
fi
if [[ $step == bench ]]; then
  python tools/pod_countdown_sft.py --model Qwen/Qwen3-8B --max_steps 20
fi
if [[ $step == train || $step == all ]]; then
  python tools/pod_countdown_sft.py --model Qwen/Qwen3-8B --out_dir "$GRAFT_COUNTDOWN_SFT_MODEL"
fi
if [[ $step == gate || $step == all ]]; then
  python tools/pod_countdown_hack_baserate.py --model "$GRAFT_COUNTDOWN_SFT_MODEL" \
    --label qwen3-8b-sft --limit 300 --k 8
  python tools/countdown_hack_baserate_grade.py --cache_dir output/countdown_hack_baserate
fi
if [[ $step == push || $step == all ]]; then
  bash "$(dirname "$0")/s3_sync.sh" push
fi
echo "RUN_SFT_OK ($step)"
