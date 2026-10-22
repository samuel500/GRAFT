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

One seed (9) to screen.

Results (2026-10-21/22, seed 9, 1xH200, eager vLLM, ~3.2 h/run; fseval n=256 at
forget scale 0 (deployed) / 1; expr-only probe n=256 at 0). Forget params =
rank-1 LoReFT 8,193/layer; retain mlp:64 = 28.3M:

  arm         forget  κ_F  fs0 comb  fs0 hack  pass@1  fs1 hack
  L20         8.2k    2    0.961     0.164     0.781   0.266   leaks into retain
  L20_k8      8.2k    8    0.840     0.000     0.840   0.430   (graft_w_max 8)
  L16_20      41k     2    0.824     0.000     0.820   0.254
  L11_20      82k     2    0.887     0.047     0.855   0.219
  Lall        295k    2    0.844     0.000     0.844   0.410
  loreft r1 L20 on both (countdown_hf100_gr_loreft.py): 0.812 / 0.047 / 0.699 / 0.488
  MLP m64 GRAFT repro (s9/1/15/16): fs0 comb 0.805-0.824, hack 0.000-0.020,
  pass@1 0.773-0.801.

Outputs: s3://<volume-id>/graft-exp/output/{exp_m64_loreft_local,
exp_m64_loreft_local_fseval,expr_only_probe_exp_m64_loreft_local}/; W&B
sknoche/small-rl runs exp_m64_loreft_local/<run_name>. EXP_RUNS=<run_name>[,...] selects a subset. Set
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

# Added 2026-10-21 after L20's in-training evals showed the hack leaking into
# retain: forget on every layer (the mixed-kind control), and L20 with the forget
# adapter's boost on detected hacks raised from 2x to 8x. κ_F = 8 needs
# graft_w_max >= 8 (the startup geometry guard; it also caps the per-coordinate
# step, a no-op at the κ operating point). κ_R is unused under classic routing.
runs += [
    {**_MIXED, "forget_layers": "all", "seed": s,
     "run_name": f"cdhf100_gr_m64_lf1_Lall_s{s}"}
    for s in (9,)
] + [
    {**_MIXED, "forget_layers": "20", "adapter_kappa": "2,8", "graft_w_max": 8.0,
     "seed": s, "run_name": f"cdhf100_gr_m64_lf1_L20_k8_s{s}"}
    for s in (9,)
]

_only = [n for n in os.environ.get("EXP_RUNS", "").split(",") if n]
if _only:
    _unknown = set(_only) - {r["run_name"] for r in runs}
    assert not _unknown, f"unknown EXP_RUNS: {sorted(_unknown)}"
    runs = [r for r in runs if r["run_name"] in _only]

per_gpu = 1
no_baseline = True
