"""No intervention WITH the GRAFT anchor and lr: the GRAFT recipe minus routing,
with nothing to ablate.

The paper's No-intervention arm (countdown_code_rp._base) differs from GRAFT in
more than routing: no LeetCode anchor, lr 5e-4 (GRAFT: 5e-4/3), opt_bs 256
(GRAFT: 272). This arm keeps GRAFT's anchor (leetcode_verified at 1:16) and lr
and batch geometry, and drops routing:

  - mlp_config m128_retain_only: ONE 128-neuron adapter. Same init and width as
    m64's two halves concatenated, but the anchor's retain-only slice now trains
    the whole adapter, i.e. the model that is evaluated (in the GRAFT-w/o-routing
    control, countdown_hf100_noroute_anchor, the anchor trains RETAIN alone while
    the eval scores both halves).
  - rh_detector_recall 0.0: the monitor never fires, so every target sample takes
    the (1, 1) path = stock GRPO advantages. Routing stays nominally "classic"
    only because coherence (the anchor slice) requires routing_mode != none.
  - renormalization_mode off, split_moment off: balanced renorm asserts both
    adapters exist; with nothing flagged both are inert anyway (split-moment ==
    stock AdamW, tests/test_split_moment_optim.py).

Eval at scale 1.0 (the whole adapter; there is no FORGET to ablate). Seeds match
the GRAFT repro (9/15/16/1). H200 settings as in repro_countdown_fig2.

    REPRO_RUNS=cdhf100_dn_lccoh64_lr3_s9 python -u sweep.py --name dn_anchor \
        --config sweeps/countdown_hf100_dn_anchor.py --no_baseline --no-eval_at_start
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
     "seed": s,
     "run_name": f"cdhf100_dn_lccoh64_lr3_s{s}"}
    for s in (9, 15, 16, 1)
]

_only = [n for n in os.environ.get("REPRO_RUNS", "").split(",") if n]
if _only:
    _unknown = set(_only) - {r["run_name"] for r in runs}
    assert not _unknown, f"unknown REPRO_RUNS: {sorted(_unknown)}"
    runs = [r for r in runs if r["run_name"] in _only]

per_gpu = 1
no_baseline = True
