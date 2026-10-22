#!/bin/bash
#   bash scripts/pod/sync.sh pull_sft|push
#   pull_sft  the SFT base from Hugging Face ($GRAFT_SFT_REPO at $GRAFT_SFT_REV) into
#             $GRAFT_COUNTDOWN_SFT_MODEL, then check the sha256 of model.safetensors
#   push      mirror rl/output/ and logs/ to $GRAFT_SYNC_URI/{output,logs/<pod>} with the
#             aws CLI; a no-op when GRAFT_SYNC_URI is unset (results stay in rl/output/)
set -euo pipefail
source "$(dirname "$0")/env.sh"
case "${1:?pull_sft|push}" in
  pull_sft)
    hf download "$GRAFT_SFT_REPO" --revision "$GRAFT_SFT_REV" --local-dir "$GRAFT_COUNTDOWN_SFT_MODEL"
    test -f "$GRAFT_COUNTDOWN_SFT_MODEL/config.json"
    echo "$GRAFT_SFT_SHA256  $GRAFT_COUNTDOWN_SFT_MODEL/model.safetensors" | sha256sum -c -
    ;;
  push)
    if [ -z "${GRAFT_SYNC_URI:-}" ]; then
      echo "GRAFT_SYNC_URI unset: results stay in $RL/output"; exit 0
    fi
    command -v aws >/dev/null || uv tool install awscli
    aws s3 sync ${GRAFT_SYNC_ARGS:-} "$RL/output/" "$GRAFT_SYNC_URI/output/" --exclude "*/__pycache__/*" --only-show-errors
    aws s3 sync ${GRAFT_SYNC_ARGS:-} "$LOGS/" "$GRAFT_SYNC_URI/logs/${RUNPOD_POD_ID:-$(hostname)}/" --only-show-errors
    ;;
esac
