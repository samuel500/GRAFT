# Countdown-Code reproductions and follow-ups (`repro` branch)

This file covers three experiments on the paper's Countdown-Code setting (Qwen3-8B):

1. A pod re-run of the paper's Fig 2 GRAFT and No-intervention arms.
2. A no-intervention anchor × learning-rate 2×2, which asks why GRAFT w/o routing hacks so much
   less than No intervention.
3. GRAFT and GRAFT w/o routing at lr 5e-4, 3× the paper's GRAFT lr.

Each section has the question, the sweep files, the run names, a rerun command, results, the
compute used, and caveats.

| What | Where |
|---|---|
| per-run eval data (28 runs: configs, per-step routing evals, n=256 fsevals, expr-only probes) | [`repro_data/`](repro_data/) |
| run index: arm, class, sweep, seed, lr, anchor, routing, adapter, deploy scale, code commit, HF path | [`repro_data/runs.csv`](repro_data/runs.csv) |
| sha256 of every file in `repro_data/` and of each run's `checkpoint-200/model.safetensors` | [`repro_data/MANIFEST.txt`](repro_data/MANIFEST.txt) |
| adapters (checkpoint-200 of every run) | HF [`samuel500/graft-repro-adapters`](https://huggingface.co/samuel500/graft-repro-adapters) (private), `<class>/<run_name>/` |
| SFT base model (frozen during RL) | HF [`samuel500/graft-qwen3-8b-countdown-sft`](https://huggingface.co/samuel500/graft-qwen3-8b-countdown-sft) (private), revision `a2d54fdf` |

All commands below run from `rl/`. The tables and the figure build offline from tracked files
only. The plotting/summary commands use `uv run --no-project` so the laptop does not sync the
full torch/vLLM env:

```bash
# all tables below (per-arm results, comparison rows, curve check)
uv run --no-project --with matplotlib --with numpy --with scipy python tools/repro_summary.py
# Fig 2 arms vs the paper's released numbers, per seed
python tools/repro_compare.py --ours_fseval repro_data/repro_fig2_fseval --ours_probe repro_data/expr_only_probe_arms
# Figure 1 with every arm of this file -> final_figures/repro_countdown_figure1_dn{,_left}.{png,pdf}
# (_left = the scatter panel on its own; also _left_no_lr5e-4, without the lr-5e-4 GRAFT / GRAFT w/o routing arms)
bash figures/build_dn_figroot.sh
uv run --no-project --with matplotlib --with numpy --with scipy python figures/repro_countdown_figure1_dn.py
# refresh repro_data/ from the network volume (only needed if runs change)
bash repro_data/fetch.sh
```

## Common setup

- **Model:** every run trains adapters on the frozen SFT base above for 200 GRPO steps
  (HF cosine schedule, 20 warmup steps). The base model is the merged Qwen3-8B Countdown-Code SFT
  model made with `scripts/pod/run_sft.sh`. On a pod, `scripts/pod/s3_sync.sh pull_sft` pulls it
  from the network volume and checks its sha256.
- **Hardware:** one 1× H200 SXM Runpod pod per seed (secure cloud, US-NC-1 / US-CO-1, image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, $5.29/h). The exception is
  `dn_anchor`, which ran 4 seeds on one pod. Runs use the H200 settings (vLLM GPU memory 0.7,
  no gradient checkpointing). A seed takes about 2.6-4 pod-hours including setup, a 4-step smoke
  test and the evals.
- **Pod flow:** `bash scripts/pod/setup.sh` (uv venv, vLLM patches, rl-rewardhacking data, awscli),
  then a queue script:
  - `scripts/pod/run_seed_queue.sh <sweep_file> <sweep_name> <run_prefix> <scales> "<seeds>"`
    runs a 4-step smoke test, then for each seed: train, n=256 fseval and expr-only probe (k=1)
    at `<scales>`, and a push of `output/` to the volume.
  - `run_rl.sh` (Fig 2) and `run_dn_anchor.sh` (`dn_anchor`) are earlier one-off versions of the
    same queue.
- **Launch pattern** (the orchestration scripts are not in the repo; they hold personal keys and
  ops-box paths): an ops box creates the pod through the Runpod REST API and copies over a
  `git bundle` of the branch, a W&B `.netrc` and the volume's S3 keys (`/root/.graft_s3`). It then
  starts `setup.sh && <queue>` in tmux, polls `logs/<sweep_name>.done`, checks that every seed's
  results reached the volume, and stops the pod.
- **Code commits:** listed per run in `runs.csv`. Training code is identical across 16844a3,
  a9f8ee1, bd53a31 and 235718f; later commits only add sweep files, queue scripts and the
  `m128_retain_only` vLLM preset.
- **Metrics** (`tools/repro_summary.py`). All are at step 200, n=256 eval problems:
  - **hack:** ground-truth hack rate (`hack_freq/`).
  - **flagged:** hacks the monitor detects (`hack_freq_detectable/`).
  - **proxy:** success under the hackable reward (`combined/`).
  - **retain:** non-hack success (`retain/`).
  - **pass1:** clean capability, i.e. expr-only pass@1 from `tools/expr_only_probe.py` with
    the hack affordance removed. This is Figure 1's y axis.
- **Scales:**
  - Scale is the FORGET adapter's multiplier, with RETAIN at 1.
  - **0.0** = RETAIN only. This is GRAFT's deployed model, and the scale at which the paper
    reports GRAFT w/o routing.
  - **1.0** = both adapters. No intervention is reported here; it has no FORGET adapter to
    ablate.
  - Means are over seeds 9/15/16/1, ± SEM.

## 1. Fig 2 reproduction

**Question:** does a fresh pod run of the released configs reproduce the paper's Countdown GRAFT
and No-intervention numbers?

- **Sweep:** `sweeps/repro_countdown_fig2.py`, sweep name `repro_fig2`. GRAFT uses `_COMMON` of
  `sweeps/countdown_hf100_gr_lccoh64_lr3.py`: LeetCode anchor 1:16, lr 5e-4/3. No intervention uses
  `_base` of `sweeps/countdown_code_rp.py`: lr 5e-4, no anchor, routing off.
- **Runs:**
  - GRAFT: `cdhf100_gr_lccoh64_lr3_s{9,15,16,1}`.
  - No intervention: `countdown_code_hack_reward_penalty_amountmissing_s{9,15,16}`. Seed 1 ran
    later as `dn_paper_s1`, see section 2.
  - Commit 16844a3, 2026-10-20.
- **Rerun** (one pod per run): `bash scripts/pod/setup.sh && bash scripts/pod/run_rl.sh <run_name>`.
  Run s9 also writes the base-model fseval (retain 0, forget 0); run s16 writes the base-model
  probe.
- **Compute:** 7 pods, each with 2.3-3.5 h of training plus setup and evals.

**Results** (`tools/repro_compare.py`; paper = its released fsevals):

| Arm (scale) | hack, ours | hack, paper | retain, ours | retain, paper | pass1, ours | pass1, paper |
|---|---|---|---|---|---|---|
| GRAFT (0.0), seeds 9/15/16/1 | 0.011 ± 0.004 | 0.017 ± 0.008 | 0.807 ± 0.004 | 0.794 ± 0.008 | 0.786 ± 0.007 | 0.716 ± 0.041 |
| No intervention (1.0), seeds 9/15/16 | 0.807 ± 0.164 | 0.792 ± 0.193 | 0.193 ± 0.164 | 0.208 ± 0.193 | 0.264 ± 0.157 | 0.234 ± 0.186 |
| Base model (SFT) | 0.074 | 0.047 | 0.562 | 0.602 | 0.562 | 0.586 |

The arm means reproduce. **Caveats:**
- Individual seeds do not reproduce: vLLM sampling is not bitwise deterministic. No intervention
  is bimodal in both runs, but with different seeds hacking: ours s9 0.94 / s16 0.48, the paper's
  s9 0.41 / s16 0.98.
- Our GRAFT pass1 is higher than the paper's (0.786 vs 0.716), mostly because the paper's s1
  (0.613) and s16 (0.691) are lower.

## 2. No intervention: anchor × learning rate

**Question:** the paper's GRAFT w/o routing control hacks at 28% (both adapters), its No
intervention at 79%. GRAFT w/o routing uses GRAFT's recipe (LeetCode anchor, lr 5e-4/3); No
intervention uses neither (lr 5e-4, no anchor). Which ingredient accounts for the gap?

All four arms are no-intervention, evaluated at scale 1.0 with nothing ablated. In the anchored
arms the anchor trains the same adapter that is evaluated: one 128-neuron adapter
(`m128_retain_only`, m64's two halves concatenated), `rh_detector_recall 0`, renorm and
split-moment off.

| Arm | Sweep file | Sweep name | Runs | Commit |
|---|---|---|---|---|
| lr 5e-4, no anchor (paper config) | `repro_countdown_fig2.py` + `countdown_dn_paper_s1.py` | `repro_fig2` + `dn_paper_s1` | `countdown_code_hack_reward_penalty_amountmissing_s{9,15,16}` + `_s1` | 16844a3, bd53a31 |
| lr/3, no anchor | `countdown_hf100_dn_lr3.py` | `dn_lr3` | `cdhf100_dn_lr3_s*` | bd53a31 |
| lr 5e-4, anchor | `countdown_hf100_dn_anchor_lr1.py` | `dn_anchor_lr1` | `cdhf100_dn_lccoh64_lr1_s*` | bd53a31 |
| lr/3, anchor | `countdown_hf100_dn_anchor.py` | `dn_anchor` | `cdhf100_dn_lccoh64_lr3_s*` | a9f8ee1 |

**Rerun** (after `setup.sh`, one pod per seed; seeds `9 15 16 1`):

```bash
bash scripts/pod/run_seed_queue.sh sweeps/countdown_hf100_dn_lr3.py dn_lr3 cdhf100_dn_lr3 1.0 "9"
bash scripts/pod/run_seed_queue.sh sweeps/countdown_hf100_dn_anchor_lr1.py dn_anchor_lr1 cdhf100_dn_lccoh64_lr1 1.0 "9"
bash scripts/pod/run_seed_queue.sh sweeps/countdown_dn_paper_s1.py dn_paper_s1 countdown_code_hack_reward_penalty_amountmissing 1.0 "1"
bash scripts/pod/run_dn_anchor.sh        # lr/3 + anchor: all 4 seeds on one pod
```

**Compute:** `dn_anchor`: 1 pod, 12.1 h. The 2×2 plus `dn_paper_s1`: 9 pods, 2.6-3.4 h each,
26.7 pod-hours. 3 more pods in EUR-IS-4 were terminated during setup because of slow downloads.

**Results** (n=256, scale 1.0, 4 seeds):

| | no anchor | anchor |
|---|---|---|
| **lr 5e-4** | hack 0.787 ± 0.118, pass1 0.271 ± 0.111 | hack 0.702 ± 0.112, pass1 0.331 ± 0.116 |
| **lr/3** | hack 0.366 ± 0.057, pass1 0.689 ± 0.059 | hack 0.419 ± 0.108, pass1 0.619 ± 0.105 |

At step 200, lr/3 accounts for essentially all of the drop. The anchor's effect is within noise
in both columns. Routing is what takes hacking to about 1% (GRAFT, section 1).

**Caveat: lr/3 mostly delays hacking.**
- Both lrs use the same cosine schedule, so a whole lr/3 run has the cumulative lr of a lr 5e-4 run
  at step 44. `repro_summary.py` compares the per-step routing evals at equal cumulative lr (n=64
  per eval, seed mean).
- Without the anchor, the lr/3 curve tracks the lr 5e-4 run's early curve. At the end of the lr/3
  run, hack is 0.33 vs 0.34 for lr 5e-4 at the matched step 44. The lr 5e-4 run then keeps rising
  to about 0.8.
- The apparent plateau at lr/3 comes from the cosine decay, not from the run converging.
- The anchored pair matches less well: lr/3 ends at 0.48 vs 0.28 at the matched step, driven by
  s15.
- So the reduction is mostly no-intervention training less, not the lower lr steering training
  elsewhere.

## 3. GRAFT and GRAFT w/o routing at lr 5e-4

**Questions:** (a) At a matched lr, does GRAFT w/o routing still hack much less than No
intervention? (b) With a 3× larger lr budget, does GRAFT still prevent hacking, or only delay it?

- **Sweeps:** `sweeps/countdown_hf100_gr_lccoh64_lr1.py` (`gr_lr1`, runs
  `cdhf100_gr_lccoh64_lr1_s*`) and `sweeps/countdown_hf100_noroute_anchor_lr1.py` (`noroute_lr1`,
  runs `cdhf100_noroute_anchor_lr1_s*`).
  - These are the Fig 2 GRAFT and the paper's GRAFT w/o routing control with only the lr changed
    (5e-4).
  - The pair differs only in `rh_detector_recall` (1.0 vs 0.0).
  - Both evaluated at scales 0.0 and 1.0. Commit 235718f.
- **Rerun** (after `setup.sh`, one pod per seed):

```bash
bash scripts/pod/run_seed_queue.sh sweeps/countdown_hf100_gr_lccoh64_lr1.py gr_lr1 cdhf100_gr_lccoh64_lr1 0.0,1.0 "9"
bash scripts/pod/run_seed_queue.sh sweeps/countdown_hf100_noroute_anchor_lr1.py noroute_lr1 cdhf100_noroute_anchor_lr1 0.0,1.0 "9"
```

**Compute:** 8 pods, 3.3-4.1 h each, except noroute s9 at 6.3 h. About 32 pod-hours.

**Results** (n=256, 4 seeds; paper rows are 3 seeds):

| Arm | Scale | hack | flagged | retain | pass1 |
|---|---|---|---|---|---|
| GRAFT, lr/3 (section 1) | 0.0 | 0.011 ± 0.004 | 0.006 | 0.807 | 0.786 ± 0.007 |
| GRAFT, lr 5e-4 | 0.0 | 0.077 ± 0.039 | 0.029 | 0.540 | 0.509 ± 0.110 |
| GRAFT, lr 5e-4 | 1.0 | 0.431 ± 0.030 | 0.431 | 0.560 | 0.565 ± 0.022 |
| GRAFT w/o routing, lr/3 (paper) | 0.0 | 0.219 ± 0.010 | 0.194 | 0.770 | 0.764 ± 0.006 |
| GRAFT w/o routing, lr/3 (paper) | 1.0 | 0.281 ± 0.019 | 0.281 | 0.718 | - |
| GRAFT w/o routing, lr 5e-4 | 0.0 | 0.411 ± 0.100 | 0.409 | 0.582 | 0.592 ± 0.079 |
| GRAFT w/o routing, lr 5e-4 | 1.0 | 0.615 ± 0.127 | 0.615 | 0.385 | 0.403 ± 0.114 |
| No intervention, lr 5e-4 (section 2) | 1.0 | 0.787 ± 0.118 | 0.787 | 0.213 | 0.271 ± 0.111 |
| Base model (SFT) | - | 0.074 | 0.020 | 0.562 | 0.562 |

- **(a) The gap is mostly lr.**
  - At lr 5e-4, GRAFT w/o routing with both adapters hacks at 0.62, close to No intervention
    (0.79) and No intervention + anchor (0.70).
  - The paper's 0.28 for this control comes from GRAFT's lr/3.
- **(b) GRAFT keeps hacking near the base rate, but loses its capability gain.**
  - Deployed GRAFT at lr 5e-4 hacks at 0.077, close to the base model's 0.074.
  - Its pass1 is 0.51, against 0.79 at lr/3 and 0.56 for the base model.
  - Per seed (hack / pass1):
    - s9: 0.09 / 0.23, collapsed.
    - s15: 0.04 / 0.77, clean.
    - s16: 0.00 / 0.56.
    - s1: 0.18 / 0.48. Most of its hacks are unflagged constraint relaxation (0.094 of 0.18).
  - The curve check shows a horizon effect here too. At equal cumulative lr, GRAFT 5e-4 RETAIN
    matches lr/3 up to about step 44 (hack 0.01-0.02, retain 0.72), which is lr/3's whole
    budget. After that, RETAIN's retain falls to about 0.50 and hacking leaks up to about 0.09.

**Caveats:**
- Four seeds with large seed-to-seed variance. The per-step curve evals are n=64.
- The paper's GRAFT w/o routing has no expr-only probe at scale 1.0.

## Before going public

Not done yet:
- `scripts/pod/env.sh` hardcodes the network volume's S3 URI, and `s3_sync.sh pull_sft` needs our
  S3 keys. Make S3 optional and let pods pull the SFT base from HF.
- Scrub infra identifiers: the volume id in `env.sh`, `repro_data/fetch.sh` and this file's
  history, and ops-box host names.
- Both HF repos are private. Their `meta.json` files point `code_repo` at the private staging repo
  (`samuel500/GRAFT`). Regenerate them with the public fork's URL when publishing.
