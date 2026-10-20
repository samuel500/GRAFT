#!/bin/bash
# Mirror pod results to the network volume:  bash scripts/pod/s3_sync.sh push|pull_sft
#   push      rl/output/ and logs/ -> $S3_URI/{output,logs}/<pod>  (logs per pod)
#   pull_sft  the merged SFT model from $S3_URI (RL pods)
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${AWS_ACCESS_KEY_ID:?no S3 keys: write /root/.graft_s3 first}"
case "${1:?push|pull_sft}" in
  push)
    aws s3 sync $S3_ARGS "$RL/output/" "$S3_URI/output/" --exclude "*/__pycache__/*" --only-show-errors
    aws s3 sync $S3_ARGS "$LOGS/" "$S3_URI/logs/${RUNPOD_POD_ID:-$(hostname)}/" --only-show-errors
    echo "local:"; du -sb "$RL/output" | cut -f1
    echo "s3:";    aws s3 ls $S3_ARGS --recursive --summarize "$S3_URI/output/" | tail -2
    ;;
  pull_sft)
    aws s3 sync $S3_ARGS "$S3_URI/output/countdown_sft_model/qwen3-8b/" "$GRAFT_COUNTDOWN_SFT_MODEL/" --only-show-errors
    aws s3 cp $S3_ARGS "$S3_URI/output/countdown_sft_model/qwen3-8b.sha256" "$GRAFT_COUNTDOWN_SFT_MODEL.sha256" --only-show-errors
    ls -la "$GRAFT_COUNTDOWN_SFT_MODEL"
    test -f "$GRAFT_COUNTDOWN_SFT_MODEL/config.json"
    # the 16 GB single-file upload reported a spurious NoSuchUpload; verify the bytes
    (cd "$GRAFT_COUNTDOWN_SFT_MODEL" && sha256sum -c "$GRAFT_COUNTDOWN_SFT_MODEL.sha256")
    ;;
esac
