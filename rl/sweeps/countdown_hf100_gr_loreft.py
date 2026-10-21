"""GRAFT on hf100 Countdown-Code with LoReFT adapters instead of MLP adapters.

Same recipe as the fig:countdown-fig1 GRAFT arm (countdown_hf100_gr_lccoh64_lr3:
LeetCode anchor 1:16, classic routing, balanced renorm + split-moment), with
both adapters swapped for LoReFT interventions on every decoder layer's output
(adapter_type='composite', gradient_routing.DualLoReFT). Equal ranks on both
sides give κ = (2, 2), the paper's doubling.

Arms: rank 1 and rank 8 (Qwen3-8B, d=4096: 8,193 / 65,544 params per adapter
per layer; 0.29M / 2.36M per adapter over 36 layers, vs 28.3M for MLP m64), at
the MLP arm's lr 5e-4/3 and at 3x that (LoReFT is usually trained at higher
lr than weight-space adapters). One seed (9) to screen.

Results (2026-10-21, seed 9, 1xH200 graft-sft, eager vLLM, ~3.2-3.8 h/run).
fseval n=256 at forget scale 0 (deployed) / 1; expr-only probe n=256 at 0;
logprob gap = W&B sampling/sampling_logp_difference/mean (median over run),
IS filter = sampling/is_seq_filter_frac (median):

  arm       fs0 comb  fs0 hack  pass@1  fs1 hack  logp gap  IS filter
  r8_lr3    0.785     0.000     0.785   0.492     0.042     5.8%
  r1_lr3    0.781     0.020     0.777   0.645     0.012     0.0%
  r8_lr1    0.488     0.031     0.391   0.969     0.218     5.9%   BROKEN
  r1_lr1    0.789     0.000     0.816   0.355     0.039     5.1%
  MLP m64 repro (s9/1/15/16): fs0 comb 0.805-0.824, hack 0.000-0.020,
  pass@1 0.773-0.801, logp gap 0.013-0.020, IS filter ~0%.

r8_lr1 is broken: deployed parse rate 0.46 (vs ~0.80), pass@1 0.39, forget-on
hacks 97%. Most likely the lr: its forget adapter drifted ~10x more than
r8_lr3's (param norm 24.0 -> 26.0 vs 24.19), train-time hack rate rose to ~0.9
(others 0.5-0.65), truncation hit 5-10% over steps 125-300.

The logp gap and IS filter columns are mostly a measurement artifact, not
LoReFT numerics. The IS filter saturates at exactly 64/1088, the coh anchor
rows: vLLM samples them at (1,0), but these runs compared them against HF old
logps taken at (1,1), before the coh slice was rescored at (1,0). So the gap
there measured the forget adapter's effect on LeetCode text, and grew with
forget-adapter drift from 0.008 at identity init. Neither the IS ratio nor
the filter reached the loss (the packed liger loss does not take
vllm_is_ratio), so no rows were dropped, and the training ratio used the
rescored (1,0) logps. Code review found the vLLM and HF LoReFT forwards
equivalent. Fixed after this screen: the coh rescore now runs before the
sampling/* metrics, which also log sampling_logp_difference/mean_{coh,routing}.
The routing-row gap is not separable in these runs; the next runs measure it.

Outputs: s3://<volume-id>/graft-exp/output/{exp_loreft,exp_loreft_fseval,
expr_only_probe_exp_loreft}/; W&B sknoche/small-rl runs exp_loreft/<run_name>.

EXP_RUNS=<run_name>[,...] selects a subset (one pod per run). Set
GRAFT_COUNTDOWN_SFT_MODEL to the SFT-primed model path.

    EXP_RUNS=cdhf100_gr_loreft_r1_lr3_s9 python -u sweep.py --name exp_loreft \
        --config sweeps/countdown_hf100_gr_loreft.py --no_baseline --no-eval_at_start
"""
import os
from sweeps.countdown_hf100_gr_lccoh64_lr3 import _COMMON

_model = os.environ.get("GRAFT_COUNTDOWN_SFT_MODEL",
                        "/workspace/graft-repro/output/countdown_sft_model/qwen3-8b")

_LOREFT = {
    **_COMMON,
    "model": _model,
    "adapter_type": "composite",
    "mlp_config": None,               # _base sets m64
    "loreft_init": "identity",
    # Composite plans are eager-only in vLLM until the compiled path is gated
    # (tools/gate_compiled_engine.py) with LoReFT layer wrappers.
    "vllm_enforce_eager": True,
}

_LRS = {"lr3": 5e-4 / 3, "lr1": 5e-4}

runs = [
    {**_LOREFT, "retain_adapter": f"loreft:{r}", "forget_adapter": f"loreft:{r}",
     "lr": lr, "seed": s, "run_name": f"cdhf100_gr_loreft_r{r}_{tag}_s{s}"}
    for r in (1, 8)
    for tag, lr in _LRS.items()
    for s in (9,)
]

_only = [n for n in os.environ.get("EXP_RUNS", "").split(",") if n]
if _only:
    _unknown = set(_only) - {r["run_name"] for r in runs}
    assert not _unknown, f"unknown EXP_RUNS: {sorted(_unknown)}"
    runs = [r for r in runs if r["run_name"] in _only]

per_gpu = 1
no_baseline = True
