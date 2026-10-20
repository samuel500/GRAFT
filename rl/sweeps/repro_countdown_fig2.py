"""Pod replication of the fig:countdown-fig1 arms: GRAFT hf100 (seeds 9/15/16/1)
and No intervention (seeds 9/15/16), one run per 1xH200 pod.

GRAFT = _COMMON of countdown_hf100_gr_lccoh64_lr3 (LeetCode anchor, lr 5e-4/3).
No intervention = _base verbatim, as in countdown_dn_s15_repro. Run names match
the original sweeps so fseval/probe files line up with paper_data. H200, so no
80 GB tweaks (vllm_gpu_memory 0.7, no gradient checkpointing).

REPRO_RUNS=<run_name>[,...] selects a subset (one pod per run). Set
GRAFT_COUNTDOWN_SFT_MODEL to the SFT-primed model path.

    REPRO_RUNS=cdhf100_gr_lccoh64_lr3_s9 python -u sweep.py --name repro_fig2 \
        --config sweeps/repro_countdown_fig2.py --no_baseline --no-eval_at_start
"""
import os
from sweeps.countdown_code_rp import _base
from sweeps.countdown_hf100_gr_lccoh64_lr3 import _COMMON

_model = os.environ.get("GRAFT_COUNTDOWN_SFT_MODEL",
                        "/workspace/graft-repro/output/countdown_sft_model/qwen3-8b")

runs = [
    {**_COMMON, "model": _model, "seed": s, "run_name": f"cdhf100_gr_lccoh64_lr3_s{s}"}
    for s in (9, 15, 16, 1)
] + [
    {**_base, "model": _model, "seed": s,
     "run_name": f"countdown_code_hack_reward_penalty_amountmissing_s{s}"}
    for s in (9, 15, 16)
]

_only = [n for n in os.environ.get("REPRO_RUNS", "").split(",") if n]
if _only:
    _unknown = set(_only) - {r["run_name"] for r in runs}
    assert not _unknown, f"unknown REPRO_RUNS: {sorted(_unknown)}"
    runs = [r for r in runs if r["run_name"] in _only]

per_gpu = 1
no_baseline = True
