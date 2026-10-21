#!/bin/bash
# One GRAFT experiment run per pod (exp/* branches): pull SFT model -> train ->
# fseval (n=256) at retain-only (forget 0.0) and both (1.0) -> expr-only probe
# at forget 0.0 -> push. The sweep file must honor EXP_RUNS.
#   bash scripts/pod/run_exp.sh <sweep_name> <sweep_file> <run_name>
#   e.g. bash scripts/pod/run_exp.sh exp_loreft sweeps/countdown_hf100_gr_loreft.py cdhf100_gr_loreft_r1_lr3_s9
set -euo pipefail
sweep=${1:?sweep_name}
config=${2:?sweep_file}
run=${3:?run_name}
source "$(dirname "$0")/env.sh"
cd "$RL"; source .venv/bin/activate
[ -f "$GRAFT_COUNTDOWN_SFT_MODEL/config.json" ] || bash "$(dirname "$0")/s3_sync.sh" pull_sft
ck=output/$sweep/$run/checkpoint-200
if [ ! -d "$ck" ]; then
  EXP_RUNS=$run python -u sweep.py --name "$sweep" --config "$config" \
    --no_baseline --no-eval_at_start --overwrite_run
fi
test -d "$ck"
python tools/local_fseval.py --sweep_dir "output/$sweep" --only_exact "$run" --n_eval 256 --scales 0.0,1.0
python tools/expr_only_probe.py --ckpt "$ck" --k 1 --n 256 --scales 0.0 --out "output/expr_only_probe_$sweep"
bash "$(dirname "$0")/s3_sync.sh" push
echo "RUN_EXP_OK $run"
