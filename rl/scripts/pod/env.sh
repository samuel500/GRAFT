# Sourced by every scripts/pod/*.sh. Pod layout: everything under /workspace
# (pod-local volume; survives a stop on the same host), results mirrored to the
# EU-RO-1 network volume <volume-id> through its S3 API.
export ROOT=/workspace/graft-repro
export RL=$ROOT/GRAFT/rl
export LOGS=$ROOT/logs
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_LINK_MODE=copy
export RH_REPO_PATH=$ROOT/ext/rl-rewardhacking
export CC_REPO=$ROOT/ext/Countdown-Code
export GRAFT_COUNTDOWN_SFT_MODEL=$RL/output/countdown_sft_model/qwen3-8b
export PYTHONUNBUFFERED=1
export PATH=$HOME/.local/bin:$PATH
export S3_URI=s3://<volume-id>/graft-repro
export S3_ARGS="--region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io"
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY for the volume's S3 API (chmod 600,
# container disk; written from <ops-host> /root/.env, never printed).
[ -f /root/.graft_s3 ] && source /root/.graft_s3
mkdir -p "$LOGS"
