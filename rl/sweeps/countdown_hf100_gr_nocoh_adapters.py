"""GRAFT on hf100 Countdown-Code WITHOUT the LeetCode anchor: MLP GRAFT and the
MLP-retain + L20 LoReFT-forget (κ_F 8) arm, repro seeds 9/15/16/1.

The two headline recipes with the anchor slice removed:
  - cdhf100_gr_nocoh_lr3_s*: paper-config MLP GRAFT (countdown_hf100_gr_lccoh64_lr3)
    minus the anchor = upstream's reviewer ablation countdown_hf100_gr_nocoh_lr3
    (same run names), here on our SFT base and with seed 1 added.
  - cdhf100_gr_m64_lf1_L20_k8_nocoh_s*: retain mlp:64 on every layer, forget
    loreft:1 on layer 20, adapter_kappa 2,8 + graft_w_max 8
    (countdown_hf100_gr_m64_loreft_local's L20_k8) minus the anchor.

No anchor = coherence at its default 'none', coh_samples_per_rollout 0 (no
coh_config, no coherence slots), optimizer batch 256 = 1024/4 (4 steps per rollout
phase, as with the anchor's 272 = 1088/4). Everything else is unchanged: classic
routing, balanced renorm + split-moment, lambda 1, lr 5e-4/3, 200 steps.

Question: how much of each arm's deployed (forget 0) retain gain and hack
suppression depends on the anchor?

One run per 1xH200 pod, all 8 in parallel (2026-10-23). Outputs:
s3://<volume-id>/graft-exp/output/{exp_nocoh,exp_nocoh_fseval,expr_only_probe_exp_nocoh}/;
W&B sknoche/small-rl runs exp_nocoh/<run_name>. EXP_RUNS=<run_name>[,...] selects
a subset. Set GRAFT_COUNTDOWN_SFT_MODEL to the SFT-primed model path.

    bash scripts/pod/run_exp_queue.sh sweeps/countdown_hf100_gr_nocoh_adapters.py \
        exp_nocoh "cdhf100_gr_nocoh_lr3_s9"
"""
import os
from sweeps.countdown_code_rp import _base
from sweeps.countdown_code_gr_nocoh import _gr  # classic/balanced/split-moment, no coherence keys

_SEEDS = (9, 15, 16, 1)

_COMMON = {
    **_base, **_gr,
    "model": os.environ.get("GRAFT_COUNTDOWN_SFT_MODEL",
                            "/workspace/graft-repro/output/countdown_sft_model/qwen3-8b"),
    "lr": 5e-4 / 3,
}

_MIXED_L20_K8 = {
    **_COMMON,
    "adapter_type": "composite",
    "mlp_config": None,               # _base sets m64; composite takes it from retain_adapter
    "retain_adapter": "mlp:64",
    "retain_layers": "all",
    "forget_adapter": "loreft:1",
    "forget_layers": "20",
    "adapter_kappa": "2,8",
    "graft_w_max": 8.0,               # κ_F 8 needs graft_w_max >= 8
    "loreft_init": "identity",
    # Composite plans are eager-only in vLLM (see countdown_hf100_gr_loreft.py).
    "vllm_enforce_eager": True,
}

runs = [
    {**_COMMON, "seed": s, "run_name": f"cdhf100_gr_nocoh_lr3_s{s}"}
    for s in _SEEDS
] + [
    {**_MIXED_L20_K8, "seed": s, "run_name": f"cdhf100_gr_m64_lf1_L20_k8_nocoh_s{s}"}
    for s in _SEEDS
]

_only = [n for n in os.environ.get("EXP_RUNS", "").split(",") if n]
if _only:
    _unknown = set(_only) - {r["run_name"] for r in runs}
    assert not _unknown, f"unknown EXP_RUNS: {sorted(_unknown)}"
    runs = [r for r in runs if r["run_name"] in _only]

per_gpu = 1
no_baseline = True
