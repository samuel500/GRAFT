# LoReFT layer-window ablations

Where along the depth do the task gain and the hacking sit in a trained
composite LoReFT GRAFT model (rank 1, retain and forget on all 36 layers of
Qwen3-8B; the `exp_loreft` r1 screen arms)?

We switch LoReFT interventions off in 5-layer windows and rerun the
post-training battery from `rl/scripts/pod/run_exp.sh` for each config:
fseval at n=256 (combined reward, retain reward, hack rate) and the expr-only
probe at k=1, n=256 (pass@1, parse rate).

Windows are inclusive decoder-layer indices, numbered from 0:
1-5, 6-10, 11-15, 16-20, 21-25, 26-30, 31-35 (`--windows` overrides). Layer 0
is in no window, so it stays on in every drop config and off in every keep
config.

| part | forget scale | configs |
|---|---|---|
| A: retain | 0 | `retain_all`, `retain_none`, and `retain_keep_W` / `retain_drop_W` per window |
| B: forget | 1 | `forget_all`, and `forget_keep_W` / `forget_drop_W` per window; retain on everywhere |

That's 31 configs. `forget_none` is the same as `retain_all`. `retain_all` and
`forget_all` should reproduce the screen's fs0 and fs1 numbers (sanity check).

A dropped intervention is zeroed in the weights (W := R, b := 0, so the edit
(h·Aᵀ + b)·R is exactly 0). This holds whatever scales the eval code sets, and
`ablate.py` asserts it per config.

The same grid also runs on MLP GRAFT checkpoints (`adapter_type: mlp`, retain
and forget mini-SwiGLUs on every layer). There a dropped adapter gets
`down_<role> := 0`, which zeroes its output. First target: the paper-config
GRAFT run `cdhf100_gr_lccoh64_lr3_s1` (m64, LeetCode anchor, lr 5e-4/3, seed 1),
pulled from `samuel500/graft-repro-adapters` (`graft/cdhf100_gr_lccoh64_lr3_s1`).
Its published deployed eval (fs0, n=256) is combined 0.824, retain 0.809,
hack 0.016, pass@1 0.797, so `retain_all` should land there.

## Files

- `ablate.py` (pod): runs the battery over the configs (or one shard of them) on
  one GPU. It resumes, skipping configs whose JSON exists. `--list` prints the grid.
- `run_pod.sh` (pod): pulls the checkpoint (a local `CKPT`, Hugging Face with
  `HF_ADAPTER`, or an S3 mirror at `GRAFT_SYNC_URI`) and the SFT model (Hugging Face,
  sha256-checked), runs a 2-config smoke test, starts one shard per GPU, then pushes
  `rl/output/` to `GRAFT_SYNC_URI` if it is set (`rl/scripts/pod/sync.sh`).
  Several pods can split the shards via `SHARD_OFFSET`/`TOTAL_SHARDS`.
- `run_queue.sh` (pod): several Hugging Face adapters back to back, one
  `run_pod.sh` per run.
- `analyze.py`: writes `summary.md`. Keep deltas are against the all-off
  reference, drop deltas against the all-on one.
- `plot.py`: draws `plots/`. `retain_windows.png` and `forget_windows.png` show
  per-window metrics against the all-on and all-off references;
  `effect_share.png` shows each window's share of the full effect.

Each `results/<run_name>/` holds:

- `summary.md`: the tables.
- `plots/`: the charts.
- `configs/<config>.json`: one per config, with the config, the fseval means,
  the probe stats and timings.
- `per_sample/<config>__fseval_values.json`: per-sample 0/1 values for every
  fseval metric (used for the SEs).
- `per_sample/<config>__probe.jsonl`: the probe's per-problem completions and
  correctness.

`summary.md` describes this layout in more detail. The `per_sample/` folders
(about 55 MB) are not in the repository, so `analyze.py` and `plot.py` need a
fresh run of the battery; `summary.md`, `plots/` and `configs/` are tracked.

## Run

On a GPU machine set up with `rl/scripts/pod/setup.sh` (`GRAFT_ROOT` = its
checkout root), from the repository root:

```bash
# MLP GRAFT seed 1, adapter + SFT base from Hugging Face, all 31 configs, one
# shard per visible GPU (SHARD_OFFSET/TOTAL_SHARDS split the grid across machines)
CKPT=/workspace/hf_adapters/cdhf100_gr_lccoh64_lr3_s1/checkpoint-200 \
  HF_ADAPTER=graft/cdhf100_gr_lccoh64_lr3_s1 \
  bash experiments/loreft_layer_ablation/run_pod.sh cdhf100_gr_lccoh64_lr3_s1

# the other MLP GRAFT seeds back to back
bash experiments/loreft_layer_ablation/run_queue.sh \
  cdhf100_gr_lccoh64_lr3_s9 cdhf100_gr_lccoh64_lr3_s15 cdhf100_gr_lccoh64_lr3_s16

# then, wherever rl/output/loreft_layer_ablation/<run>/ was copied to:
python experiments/loreft_layer_ablation/analyze.py <results>/<run>
python experiments/loreft_layer_ablation/plot.py <results>/<run>
```

The LoReFT run (`cdhf100_gr_loreft_r1_lr3_s9`) used a local checkpoint
(`CKPT`). Its adapter is in `samuel500/graft-loreft-adapters`.

Cost: the screen's battery ran about 8-9 min per eval pass on one H200 (HF
generate). That makes 31 configs x 2 passes about 8-9 GPU-hours, split across
however many GPUs the pod has.
