"""GRAFT at the paper's No-intervention lr (5e-4): does routing prevent hacking
or only delay it?

At GRAFT's lr (5e-4/3) No intervention ends where its lr-5e-4 run was at step
~45 (equal cumulative lr under the cosine schedule), so lr/3 mostly slows it
down. This arm gives GRAFT the same 3x budget: _COMMON of
countdown_hf100_gr_lccoh64_lr3 verbatim except lr. Eval at scale 0.0 (RETAIN
only, the deployed model) and 1.0 (pre-ablation). Seeds match the GRAFT repro
(9/15/16/1). H200 settings as in repro_countdown_fig2.

    REPRO_RUNS=cdhf100_gr_lccoh64_lr1_s9 python -u sweep.py --name gr_lr1 \
        --config sweeps/countdown_hf100_gr_lccoh64_lr1.py --no_baseline --no-eval_at_start
"""
import os
from sweeps.countdown_hf100_gr_lccoh64_lr3 import _COMMON

_model = os.environ.get("GRAFT_COUNTDOWN_SFT_MODEL",
                        "/workspace/graft-repro/output/countdown_sft_model/qwen3-8b")

runs = [
    {**_COMMON, "model": _model, "lr": 5e-4, "seed": s,
     "run_name": f"cdhf100_gr_lccoh64_lr1_s{s}"}
    for s in (9, 15, 16, 1)
]

_only = [n for n in os.environ.get("REPRO_RUNS", "").split(",") if n]
if _only:
    _unknown = set(_only) - {r["run_name"] for r in runs}
    assert not _unknown, f"unknown REPRO_RUNS: {sorted(_unknown)}"
    runs = [r for r in runs if r["run_name"] in _only]

per_gpu = 1
no_baseline = True
