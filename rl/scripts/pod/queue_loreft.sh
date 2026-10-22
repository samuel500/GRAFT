#!/bin/bash
# Queue the 4 LoReFT screen arms (seed 9) back to back from the exp checkout at
# /workspace/graft-exp, reusing the repro checkout's ext repos, venv (symlink
# GRAFT/rl/.venv -> /workspace/graft-repro/GRAFT/rl/.venv) and SFT model.
# Waits for the smoke-loreft tmux session and aborts unless it passed.
# Does NOT stop the pod at the end; the idle watcher catches it after 1 h.
#   tmux new -d -s queue-loreft "bash queue_loreft.sh > logs/queue_loreft.log 2>&1"
set -uo pipefail
export GRAFT_ROOT=/workspace/graft-exp GRAFT_EXT=/workspace/graft-repro/ext
export GRAFT_COUNTDOWN_SFT_MODEL=/workspace/graft-repro/GRAFT/rl/output/countdown_sft_model/qwen3-8b
log=/workspace/graft-exp/logs/smoke_loreft.log
while tmux has-session -t "=smoke-loreft" 2>/dev/null; do sleep 30; done
grep -q SMOKE_LOREFT_OK "$log" || { echo "QUEUE_ABORT: smoke did not pass"; exit 1; }
cd /workspace/graft-exp/GRAFT/rl
for run in cdhf100_gr_loreft_r8_lr3_s9 cdhf100_gr_loreft_r1_lr3_s9 cdhf100_gr_loreft_r8_lr1_s9 cdhf100_gr_loreft_r1_lr1_s9; do
  echo "QUEUE_START $run $(date -u +%FT%TZ)"
  bash scripts/pod/run_exp.sh exp_loreft sweeps/countdown_hf100_gr_loreft.py $run \
    > /workspace/graft-exp/logs/$run.log 2>&1 && echo "QUEUE_DONE $run $(date -u +%FT%TZ)" \
    || echo "QUEUE_FAIL $run $(date -u +%FT%TZ)"
done
echo QUEUE_ALL_DONE
