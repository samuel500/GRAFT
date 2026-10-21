"""Anchored no intervention at the paper's No-intervention lr: the anchor-only
cell of the anchor x lr 2x2 (see countdown_hf100_dn_lr3.py for the grid).

countdown_hf100_dn_anchor verbatim (LeetCode anchor 1:16, opt_bs 272, one
128-neuron adapter, monitor off) except lr 5e-4 instead of 5e-4/3. Eval at
scale 1.0. Seeds 9/15/16/1.

    REPRO_RUNS=cdhf100_dn_lccoh64_lr1_s9 python -u sweep.py --name dn_anchor_lr1 \
        --config sweeps/countdown_hf100_dn_anchor_lr1.py --no_baseline --no-eval_at_start
"""
import os
from sweeps.countdown_hf100_gr_lccoh64_lr3 import _COMMON

_model = os.environ.get("GRAFT_COUNTDOWN_SFT_MODEL",
                        "/workspace/graft-repro/output/countdown_sft_model/qwen3-8b")

runs = [
    {**_COMMON,
     "model": _model,
     "mlp_config": "m128_retain_only",
     "rh_detector_recall": 0.0,
     "renormalization_mode": "off",
     "split_moment": False,
     "lr": 5e-4,
     "seed": s,
     "run_name": f"cdhf100_dn_lccoh64_lr1_s{s}"}
    for s in (9, 15, 16, 1)
]

_only = [n for n in os.environ.get("REPRO_RUNS", "").split(",") if n]
if _only:
    _unknown = set(_only) - {r["run_name"] for r in runs}
    assert not _unknown, f"unknown REPRO_RUNS: {sorted(_unknown)}"
    runs = [r for r in runs if r["run_name"] in _only]

per_gpu = 1
no_baseline = True
