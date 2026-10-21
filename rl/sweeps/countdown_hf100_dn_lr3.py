"""No intervention at GRAFT's lr, nothing else changed: the lr-only cell of the
anchor x lr 2x2 around the paper's No-intervention arm.

  - countdown_code_rp._base           no anchor, lr 5e-4   (paper No intervention)
  - THIS FILE                         no anchor, lr 5e-4/3
  - countdown_hf100_dn_anchor_lr1     anchor,    lr 5e-4
  - countdown_hf100_dn_anchor         anchor,    lr 5e-4/3

_base verbatim (dual m64 adapters, routing none, opt_bs 256) except lr. Eval at
scale 1.0, as for the paper arm. Seeds match the GRAFT repro (9/15/16/1).

    REPRO_RUNS=cdhf100_dn_lr3_s9 python -u sweep.py --name dn_lr3 \
        --config sweeps/countdown_hf100_dn_lr3.py --no_baseline --no-eval_at_start
"""
import os
from sweeps.countdown_code_rp import _base

_model = os.environ.get("GRAFT_COUNTDOWN_SFT_MODEL",
                        "/workspace/graft-repro/output/countdown_sft_model/qwen3-8b")

runs = [
    {**_base, "model": _model, "lr": 5e-4 / 3, "seed": s, "run_name": f"cdhf100_dn_lr3_s{s}"}
    for s in (9, 15, 16, 1)
]

_only = [n for n in os.environ.get("REPRO_RUNS", "").split(",") if n]
if _only:
    _unknown = set(_only) - {r["run_name"] for r in runs}
    assert not _unknown, f"unknown REPRO_RUNS: {sorted(_unknown)}"
    runs = [r for r in runs if r["run_name"] in _only]

per_gpu = 1
no_baseline = True
