"""Paper No-intervention arm, seed 1: countdown_code_rp._base verbatim (as in
countdown_dn_s15_repro / repro_countdown_fig2), adding the fourth seed the GRAFT
repro uses so every cell of the anchor x lr 2x2 has seeds 9/15/16/1.

    REPRO_RUNS=countdown_code_hack_reward_penalty_amountmissing_s1 python -u sweep.py \
        --name dn_paper_s1 --config sweeps/countdown_dn_paper_s1.py --no_baseline --no-eval_at_start
"""
import os
from sweeps.countdown_code_rp import _base

_model = os.environ.get("GRAFT_COUNTDOWN_SFT_MODEL",
                        "/workspace/graft-repro/output/countdown_sft_model/qwen3-8b")

runs = [{**_base, "model": _model, "seed": 1,
         "run_name": "countdown_code_hack_reward_penalty_amountmissing_s1"}]

_only = [n for n in os.environ.get("REPRO_RUNS", "").split(",") if n]
if _only:
    _unknown = set(_only) - {r["run_name"] for r in runs}
    assert not _unknown, f"unknown REPRO_RUNS: {sorted(_unknown)}"
    runs = [r for r in runs if r["run_name"] in _only]

per_gpu = 1
no_baseline = True
