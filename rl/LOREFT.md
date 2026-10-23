# GRAFT with LoReFT adapters (Countdown-Code, `loreft` branch)

Follow-ups to the Countdown-Code reproduction in [REPRO.md](REPRO.md) (Qwen3-8B, same SFT
base, recipe and eval protocol):

1. GRAFT with LoReFT adapters: LoReFT on both sides, and an MLP RETAIN adapter with a small,
   layer-local LoReFT FORGET adapter.
2. MLP GRAFT and the best MLP + LoReFT arm without the LeetCode anchor.

Each section has the question, the sweep files, the run names, a rerun command, results, the
compute used, and caveats. Two related analyses have their own READMEs:
[`experiments/loreft_layer_ablation/`](../experiments/loreft_layer_ablation/) (which layers carry
the RETAIN gain and the FORGET hacking, by switching adapters off in 5-layer windows) and
[`experiments/steering_variance/`](../experiments/steering_variance/) (the paper's PPS steering
vector against natural layer-20 activations and our LoReFT FORGET directions).

| What | Where |
|---|---|
| per-run eval data (26 runs, sweeps `exp_loreft`, `exp_m64_loreft_local`, `exp_nocoh`) | [`repro_data/`](repro_data/), same layout as REPRO.md's runs |
| run index (classes `loreft`, `mlp-loreft`, and `graft` with anchor `none`) | [`repro_data/runs.csv`](repro_data/runs.csv) |
| sha256 of the tracked files and of each run's `checkpoint-200/model.safetensors` | [`repro_data/MANIFEST.txt`](repro_data/MANIFEST.txt) |
| adapters (checkpoint-200 of every run) | HF [`samuel500/graft-loreft-adapters`](https://huggingface.co/samuel500/graft-loreft-adapters), `<class>/<run_name>/` (`hf_path` in `runs.csv`) |
| SFT base model (frozen during RL) | HF [`samuel500/graft-qwen3-8b-countdown-sft`](https://huggingface.co/samuel500/graft-qwen3-8b-countdown-sft), revision `a2d54fdf` |

From `rl/`, offline from tracked files only:

```bash
# per-arm tables (every runs.csv arm, these included)
uv run --no-project --with matplotlib --with numpy --with scipy python tools/repro_summary.py
# -> final_figures/adapters_countdown_{figure1,figure1_left,variants,noanchor}.{png,pdf}
# (reads repro_data/ directly; PNGs copied to repro_figures/)
uv run --no-project --with matplotlib --with numpy --with scipy python figures/adapters_countdown_figure1.py
```

## 1. GRAFT with LoReFT adapters

**Questions:** (a) Does GRAFT work with LoReFT (a rank-r edit of a decoder layer's output,
Wu et al. 2024) instead of MLP adapters? (b) How small can the FORGET adapter be? Here it is a
rank-1 LoReFT on a few layers next to the paper's 28.3M-parameter MLP RETAIN adapter.

- **Code:** `--adapter_type composite` with `--retain_adapter` / `--forget_adapter`
  (`mlp:<neurons>`, `loreft:<rank>` or `none`) and `--retain_layers` / `--forget_layers`
  (e.g. `all`, `16-20`, `20`). LoReFT is `gradient_routing.DualLoReFT` in the trainer and
  `vllm_mlp_adapter.VLLMDualLoReFT` in vLLM; composite plans need eager vLLM. Mixed kinds have
  no size-derived κ, so they set `--adapter_kappa κ_R,κ_F`. Classic routing only uses κ_F, the
  FORGET adapter's gradient multiplier on detected hacks. κ_F > 4 also needs
  `--graft_w_max ≥ κ_F`.
- **Recipe:** otherwise the Fig 2 GRAFT arm of [REPRO.md](REPRO.md) section 1 (`_COMMON` of
  `countdown_hf100_gr_lccoh64_lr3.py`: LeetCode anchor 1:16, classic routing, balanced renorm +
  split-moment), lr 5e-4/3 unless noted.
- **Sizes** (Qwen3-8B, d=4096, 36 layers): MLP m64 is 786,432 params per layer (28.3M over all
  layers). Rank-1 LoReFT is 8,193 params per layer, rank 8 is 65,544.
- **Sweeps:** `sweeps/countdown_hf100_gr_loreft.py` (`exp_loreft`, LoReFT on both sides) and
  `sweeps/countdown_hf100_gr_m64_loreft_local.py` (`exp_m64_loreft_local`, MLP RETAIN on every
  layer + rank-1 LoReFT FORGET). Run names, seeds and commits are in `runs.csv` (classes
  `loreft` and `mlp-loreft`).
- **Code commits:** 295065c (the LoReFT screen), then 96c2839, e8c2b2d, 0fbd571 and 38cca0b.
  758d3cd, between the screen and the rest, moved the coherence-row logprob rescore ahead of the
  `sampling/*` IS diagnostics and refactored the probe; it is meant to change metrics only. The
  later commits add sweep entries and queue scripts. Before publication this branch's history
  was rewritten to remove infrastructure details (storage ids, host names, ops-only scripts,
  per-sample ablation outputs); the run code is unchanged, and the hashes here and in
  `runs.csv` are the rewritten ones.
- **Rerun** (after `setup.sh`; any number of runs per pod, smoke test on the first):

```bash
bash scripts/pod/run_exp_queue.sh sweeps/countdown_hf100_gr_m64_loreft_local.py exp_m64_loreft_local \
    "cdhf100_gr_m64_lf1_L20_k8_s9 cdhf100_gr_m64_lf1_L16_20_s9"
```

  `run_exp_queue.sh` trains each run, then runs the n=256 fseval at scales 0.0 and 1.0 and the
  expr-only probe at 0.0. Lines appended to `logs/<sweep_name>.runs` while it runs are queued
  too. The LoReFT screen used the earlier `run_exp.sh` / `queue_loreft.sh`.
- **Compute:** 1× H200 pods, set up as in REPRO.md's Common setup. The screen took 4 runs at
  3.2-3.8 h on one pod. Six mixed and LoReFT runs then ran on one pod in 19.8 h, and the
  remaining 7 κ_F 8 seeds plus κ_F 4 on four pods in 28 pod-hours. Total about 62 pod-hours.

**Results** (n=256, seed 9 unless noted; deployed = scale 0.0, RETAIN only; "both" = hack with
FORGET on):

| Arm | RETAIN / FORGET | κ_F | deployed hack | deployed retain | pass1 | both: hack |
|---|---|---|---|---|---|---|
| GRAFT, lr/3 (REPRO.md section 1, 4 seeds) | m64 / m64 | 2 | 0.011 ± 0.004 | 0.807 ± 0.004 | 0.786 ± 0.007 | |
| LoReFT r8, all layers | 65.5k×36 / same | 2 | 0.000 | 0.785 | 0.785 | 0.492 |
| LoReFT r1, all layers | 8.2k×36 / same | 2 | 0.020 | 0.762 | 0.777 | 0.645 |
| LoReFT r8, all layers, lr 5e-4 | 65.5k×36 / same | 2 | 0.031 | 0.457 | 0.391 | 0.969 |
| LoReFT r1, all layers, lr 5e-4 | 8.2k×36 / same | 2 | 0.000 | 0.789 | 0.816 | 0.355 |
| LoReFT r1, layer 20 only | 8.2k / 8.2k | 2 | 0.047 | 0.766 | 0.699 | 0.488 |
| MLP + LoReFT FORGET, all layers | 28.3M / 295k | 2 | 0.000 | 0.844 | 0.844 | 0.410 |
| MLP + LoReFT FORGET, layers 11-20 | 28.3M / 82k | 2 | 0.047 | 0.840 | 0.855 | 0.219 |
| MLP + LoReFT FORGET, layers 16-20 | 28.3M / 41k | 2 | 0.000 | 0.824 | 0.820 | 0.254 |
| MLP + LoReFT FORGET, layer 20 | 28.3M / 8.2k | 2 | 0.164 | 0.797 | 0.781 | 0.266 |
| MLP + LoReFT FORGET, layer 20 | 28.3M / 8.2k | 4 | 0.344 | 0.656 | 0.707 | 0.312 |
| **MLP + LoReFT FORGET, layer 20 (8 seeds)** | 28.3M / 8.2k | 8 | **0.013 ± 0.013** | **0.833 ± 0.004** | **0.833 ± 0.003** | 0.379 ± 0.037 |
| Base model (SFT) | | | 0.074 | 0.562 | 0.562 | |

The windows 11-20 and 16-20 come from `experiments/loreft_layer_ablation/`. Ablating layer
windows of the r1 all-layer run there put its FORGET adapter's hacking in layers 11-20.

- **(a) LoReFT works on both sides** at lr/3: rank 1 and rank 8 both deploy near zero hacking.
  Their pass1 matches MLP GRAFT. Rank 8 at lr 5e-4 broke: its FORGET drifted, hacked 0.97 with
  both adapters on, and the deployed model lost its capability.
- **(b) A very small FORGET adapter is enough, given enough κ_F.**
  - With an MLP RETAIN, a rank-1 LoReFT FORGET on all layers or on layers 16-20 deploys clean.
    Its pass1 (0.82-0.84) is above GRAFT's.
  - On layer 20 alone (8,193 params, 1/3,456 of RETAIN) at κ_F 2, hacking leaks into RETAIN
    (0.164). κ_F 4 is worse (0.344).
  - At κ_F 8 the same adapter holds across 8 seeds: hack 0.013 ± 0.013, pass1 0.833 ± 0.003,
    against MLP GRAFT's 0.786 ± 0.007 (the paper's 8 seeds: 0.716 ± 0.041).
- **Figure:** `adapters_countdown_figure1` (± SEM, like REPRO.md's figures). Both base models are
  shown: ours filled, the paper's released one hollow. The "+" markers are GRAFT and the κ_F 8
  arm trained without the anchor (section 2). On the right are the κ_F 8 runs'
  per-step evals.
  Deployed hacking drops to about 0 by step 60, while FORGET-only hacking rises to about 0.3 and
  its legitimate solve rate falls to about 0.35.

![Figure 1, LoReFT FORGET](repro_figures/adapters_countdown_figure1.png)

![All LoReFT variants](repro_figures/adapters_countdown_variants.png)

**Caveats:**
- Every arm except κ_F 8 is one seed.
- The κ_F 2 / 4 / 8 ordering at layer 20 is not monotone (0.164, 0.344, 0.000 at seed 9). Read it
  as a threshold, crossed somewhere at or above κ_F 8, not as a dose-response.
- κ_F 8 seed 3 deploys 0.102 hack, of which 0.082 is unflagged constraint relaxation. Routing
  cannot act on hacks the monitor misses; the other 7 seeds deploy 0.000-0.004.
- The right panels end at step 192 (no posthoc step-200 n=64 evals). Their step-0 anchor is the
  base model's n=256 fseval.
- AdamW weight decay (0.1) pulls LoReFT's W toward 0, not toward its no-op point W = R. Over these
  runs that is about a 0.2% shrink, negligible next to the learned change. It would matter for
  longer or higher-lr runs.
- The adapters of both sections are in `samuel500/graft-loreft-adapters`. Their sha256 are in
  `MANIFEST.txt` and in each run's `meta.json`.

## 2. Without the LeetCode anchor

**Question:** how much do MLP GRAFT and the κ_F 8 layer-20 arm depend on the LeetCode anchor?

- **Recipe:** both arms as in section 1, minus the anchor slice: `coherence` at its default `none`,
  `coh_samples_per_rollout` 0, no `coh_config`, optimizer batch 256 (= 1024/4, still 4 steps per
  rollout). Nothing else changes. The MLP arm is the same config as upstream's reviewer ablation
  `sweeps/countdown_hf100_gr_nocoh_lr3.py`.
- **Sweep:** `sweeps/countdown_hf100_gr_nocoh_adapters.py` (`exp_nocoh`), seeds 9/15/16/1, code
  commit 87f234c. Run names are in `runs.csv` (arms `GRAFT (lr/3), no anchor` and
  `MLP + LoReFT forget, L20, κ_F 8, no anchor`).
- **Compute:** 8 runs, one 1× H200 pod each, all in parallel: ~3.0 h per MLP pod and ~3.4 h per
  mixed pod, about 26 pod-hours.

**Results** (n=256, mean ± SEM over seeds 9/15/16/1; the anchored rows are the same 4 seeds):

| Arm | anchor | deployed hack | flagged | deployed retain | pass1 | both: hack |
|---|---|---|---|---|---|---|
| GRAFT, lr/3 | LeetCode | 0.011 ± 0.004 | 0.006 ± 0.003 | 0.807 ± 0.004 | 0.786 ± 0.007 | |
| GRAFT, lr/3 | none | 0.032 ± 0.012 | 0.008 ± 0.003 | 0.763 ± 0.016 | 0.769 ± 0.022 | 0.307 ± 0.018 |
| MLP + LoReFT FORGET, layer 20, κ_F 8 | LeetCode | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.834 ± 0.006 | 0.840 ± 0.003 | 0.364 ± 0.033 |
| MLP + LoReFT FORGET, layer 20, κ_F 8 | none | 0.004 ± 0.004 | 0.000 ± 0.000 | 0.838 ± 0.006 | 0.847 ± 0.009 | 0.290 ± 0.047 |

Per seed, deployed hack / pass1: GRAFT no anchor s9 0.004/0.816, s15 0.043/0.785, s16
0.059/0.758, s1 0.020/0.715; κ_F 8 no anchor s9 0.000/0.863, s15 0.016/0.824, s16 0.000/0.840,
s1 0.000/0.859.

- **MLP GRAFT is worse without the anchor.** Deployed retain drops by about 0.04, the seeds spread
  about 4× wider, and deployed hacking triples to 0.032. Most of that hacking is unflagged
  (0.008 flagged), so routing could not have caught it. Seed 9 is as good as the anchored runs;
  15, 16 and 1 are not.
- **The κ_F 8 layer-20 arm does not need the anchor.** Deployed hack, retain and pass1 are all
  within noise of the anchored runs. Its pass1 (0.847) stays above anchored MLP GRAFT (0.786).
- **Figure:** `adapters_countdown_noanchor`. Left, the scatter (hollow = no anchor; the paper's
  base model is not shown here). Right, the deployed (RETAIN only) per-step evals of all four
  arms (n=64 per eval, seeds faint). MLP GRAFT without the anchor drifts below the anchored
  curve on retain from about step 100, and its deployed hacking climbs back toward 5% late in
  training. The κ_F 8 curves overlap. Section 1's Figure 1 and the variants scatter
  also show both no-anchor arms ("+" markers).

![No anchor](repro_figures/adapters_countdown_noanchor.png)

**Caveats:**
- 4 seeds per arm. The MLP gap (retain 0.807 vs 0.763) is about 2.7 combined SEMs; the κ_F 8
  arms are indistinguishable.
- The seed fixes the prompt order but not adapter init or vLLM sampling (as in every run here),
  so equal seeds with and without the anchor are not paired runs.
- One of the 8 pods (MLP s9) could not push to the volume through Runpod's S3 gateway; its
  outputs were copied off the pod and uploaded from elsewhere. The uploaded size matches the pod's
  and the checkpoint sha256 matches the copy.
