"""GRAFT on hf100 Countdown-Code: MLP retain adapter, layer-localized LoReFT forget.

Same recipe as the fig:countdown-fig1 GRAFT arm (countdown_hf100_gr_lccoh64_lr3:
LeetCode anchor 1:16, classic routing, balanced renorm + split-moment, lr
5e-4/3). The retain side is the paper's MLP adapter (64 neurons on every
layer); the forget side is a rank-1 LoReFT on a few middle layers only.

Why these windows: in the layer-window ablations of the all-layer r1_lr3 LoReFT
run (experiments/loreft_layer_ablation/), the forget adapter's hacking was
localized in layers 11-20 (keep-only 16-20 hack 0.344, 11-15 0.195, the rest
near baseline; all 0.645). Arms, run in this order: layer 20 alone, 16-20,
11-20.

κ: mixed kinds have no size-derived κ, so adapter_kappa is fixed at the
paper's doubling. Under classic routing only κ_F is used (the forget adapter
steps at 2x on detected hacks); κ_R is unused.

One seed (9) to screen. EXP_RUNS=<run_name>[,...] selects a subset. Set
GRAFT_COUNTDOWN_SFT_MODEL to the SFT-primed model path.

    bash scripts/pod/run_exp_queue.sh sweeps/countdown_hf100_gr_m64_loreft_local.py \
        exp_m64_loreft_local "cdhf100_gr_m64_lf1_L20_s9 ..."
"""
import os
from sweeps.countdown_hf100_gr_lccoh64_lr3 import _COMMON

_model = os.environ.get("GRAFT_COUNTDOWN_SFT_MODEL",
                        "/workspace/graft-repro/output/countdown_sft_model/qwen3-8b")

_MIXED = {
    **_COMMON,
    "model": _model,
    "adapter_type": "composite",
    "mlp_config": None,               # _base sets m64; composite takes it from retain_adapter
    "retain_adapter": "mlp:64",
    "retain_layers": "all",
    "forget_adapter": "loreft:1",
    "adapter_kappa": "2,2",
    "loreft_init": "identity",
    # Composite plans are eager-only in vLLM (see countdown_hf100_gr_loreft.py).
    "vllm_enforce_eager": True,
}

_WINDOWS = {"L20": "20", "L16_20": "16-20", "L11_20": "11-20"}

runs = [
    {**_MIXED, "forget_layers": layers, "seed": s,
     "run_name": f"cdhf100_gr_m64_lf1_{tag}_s{s}"}
    for tag, layers in _WINDOWS.items()
    for s in (9,)
]

_only = [n for n in os.environ.get("EXP_RUNS", "").split(",") if n]
if _only:
    _unknown = set(_only) - {r["run_name"] for r in runs}
    assert not _unknown, f"unknown EXP_RUNS: {sorted(_unknown)}"
    runs = [r for r in runs if r["run_name"] in _only]

per_gpu = 1
no_baseline = True
