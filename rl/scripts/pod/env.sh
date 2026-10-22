# Sourced by every scripts/pod/*.sh. Pod layout: everything under /workspace
# (the pod's volume; survives a stop on the same host).
export ROOT=/workspace/graft-repro
export RL=$ROOT/GRAFT/rl
export LOGS=$ROOT/logs
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_LINK_MODE=copy
export RH_REPO_PATH=$ROOT/ext/rl-rewardhacking
export CC_REPO=$ROOT/ext/Countdown-Code
export GRAFT_COUNTDOWN_SFT_MODEL=$RL/output/countdown_sft_model/qwen3-8b
# The SFT base (frozen during RL): pulled by sync.sh pull_sft, sha256 of model.safetensors
export GRAFT_SFT_REPO=samuel500/graft-qwen3-8b-countdown-sft
export GRAFT_SFT_REV=a2d54fdf0aa9fa6ee8e487777ee7d63adaf8ec48
export GRAFT_SFT_SHA256=533c4f2b03dce7f21ae655a522ceb30d6781aa158fe4c486565491d3a0bd2904
export PYTHONUNBUFFERED=1
export PATH=$HOME/.local/bin:$PATH
# Optional results mirror (sync.sh push), off unless set, e.g. in ~/.graft_sync:
#   GRAFT_SYNC_URI=s3://<bucket>/graft-repro   GRAFT_SYNC_ARGS="--endpoint-url ..."
#   plus AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
[ -f "$HOME/.graft_sync" ] && source "$HOME/.graft_sync"
mkdir -p "$LOGS"
