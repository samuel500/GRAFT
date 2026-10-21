"""The paper's GRAFT w/o routing control (countdown_hf100_noroute_anchor) at the
No-intervention lr (5e-4): its runs verbatim except lr, with the H100 box
tweaks (_BOX) replaced by GRAFT's H200 values (vllm_gpu_memory 0.7, no gradient
checkpointing). Pairs with countdown_hf100_gr_lccoh64_lr1: the two differ only
in rh_detector_recall (routing on vs off at lr 5e-4).

Eval at scale 0.0 (RETAIN only) and 1.0 (both adapters), the two numbers the
paper reports for this control. Seeds 9/15/16 as in the paper, plus 1.

    REPRO_RUNS=cdhf100_noroute_anchor_lr1_s9 python -u sweep.py --name noroute_lr1 \
        --config sweeps/countdown_hf100_noroute_anchor_lr1.py --no_baseline --no-eval_at_start
"""
import os
from sweeps.countdown_hf100_noroute_anchor import runs as _paper_runs

_model = os.environ.get("GRAFT_COUNTDOWN_SFT_MODEL",
                        "/workspace/graft-repro/output/countdown_sft_model/qwen3-8b")
_r = {k: v for k, v in _paper_runs[0].items()
      if k not in ("vllm_gpu_memory", "gradient_checkpointing", "seed", "run_name")}

runs = [
    {**_r, "model": _model, "lr": 5e-4, "vllm_gpu_memory": 0.7,
     "gradient_checkpointing": False, "seed": s,
     "run_name": f"cdhf100_noroute_anchor_lr1_s{s}"}
    for s in (9, 15, 16, 1)
]

_only = [n for n in os.environ.get("REPRO_RUNS", "").split(",") if n]
if _only:
    _unknown = set(_only) - {r["run_name"] for r in runs}
    assert not _unknown, f"unknown REPRO_RUNS: {sorted(_unknown)}"
    runs = [r for r in runs if r["run_name"] in _only]

per_gpu = 1
no_baseline = True
