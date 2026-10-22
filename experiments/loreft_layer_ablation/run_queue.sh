#!/bin/bash
# Several Hugging Face adapters back to back on one pod: run_pod.sh per run
# (adapter graft/<run> from samuel500/graft-repro-adapters), carrying on past
# a failed run. Run after rl/scripts/pod/setup.sh. Per-run logs in
# $LOGS/ablate_<run>.log, one line per run in $LOGS/ablate_queue.log, and
# $LOGS/ablate_queue.done (QUEUE_OK / QUEUE_FAIL ...) at the end.
#   bash experiments/loreft_layer_ablation/run_queue.sh cdhf100_gr_lccoh64_lr3_s9 cdhf100_gr_lccoh64_lr3_s15
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
LOGS=${GRAFT_ROOT:-/workspace/graft-exp}/logs
mkdir -p "$LOGS"; rm -f "$LOGS/ablate_queue.done"
bad=()
for run in "$@"; do
  rm -f "$LOGS/ablate.done"
  echo "QUEUE_RUN $run $(date -u +%FT%TZ)" >> "$LOGS/ablate_queue.log"
  CKPT=/workspace/hf_adapters/$run/checkpoint-200 HF_ADAPTER=${HF_PREFIX:-graft}/$run \
    bash "$here/run_pod.sh" "$run" > "$LOGS/ablate_$run.log" 2>&1
  r=$(cat "$LOGS/ablate.done" 2>/dev/null || echo "ABLATE_FAIL $run run_pod died")
  echo "$r" >> "$LOGS/ablate_queue.log"
  case "$r" in ABLATE_OK*) ;; *) bad+=("$run") ;; esac
done
if [ ${#bad[@]} = 0 ]; then r="QUEUE_OK $# runs"; else r="QUEUE_FAIL ${bad[*]}"; fi
echo "$r $(date -u +%FT%TZ)" | tee "$LOGS/ablate_queue.done"
