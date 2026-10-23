# Steering variance: fixed PPS vs natural activations

Positive preventative steering (PPS) is the upstream baseline that adds one
fixed vector, α·v_L (best arm: `hackonly`, layer 20, α = 2), to the residual
stream during RL training. The vector is added at every position of every
training sequence: vLLM prefill and decode, and the HF update forward, prompt
included. Eval, deployment and the KL reference are unsteered.

This folder asks how that constant shift compares with how realistic layer-20
activations actually move along the same direction, token by token, in rollouts
that hack and rollouts that don't. It also asks how the PPS direction relates
to what our LoReFT forget adapters learn.

All numbers below are from the SFT-base replay (`results/plots/base/`). Replaying
through the no-intervention checkpoint (`results/plots/dn/`) gives nearly the
same values (per-token d′ 1.39 vs 1.41, rollout-mean d′ 9.64 vs 9.75): RL
training did not move this axis at layer 20.

## How the data was made

`steering_variance.py` runs in phases on one GPU (an H200 here). `run_pod.sh`
runs them in order on a machine set up with `rl/scripts/pod/setup.sh`.

1. **`pps`: the steering vector.** This is upstream's `tools/modal_pps_extract.py::extract`
   without Modal. The constants and prompts are read from that file with `ast`,
   so the recipe is identical. The SFT base generates on 128 train-split
   questions (half hackable) with 3 "rewrite `verify_solution`" system suffixes
   and 3 anti-hack suffixes. A filter keeps positives that tampered and
   negatives that didn't. The vector is the difference of response-token means
   (positive minus negative) at each layer's output. Upstream's layer/α
   selection sweep is skipped, because the paper already fixed `hackonly`, L20,
   α = 2. Result: ‖v₂₀‖ = 23.1, so **the PPS shift along the unit direction is
   α‖v₂₀‖ = 46.2**. For scale, the mean residual norm at layer 20 is about 115.
2. **`rollouts`: natural behaviour.** Rollouts use the neutral training prompt
   on 128 all-hackable eval-split questions × 8 samples, at temperature 1 with
   no top-k (the training sampler). They come from two policies:
   - the SFT base (test_overwrite rate 2.3%);
   - the no-intervention RL checkpoint `cdhf100_dn_lr3_s9` from
     `samuel500/graft-repro-adapters`, at deploy scales (1, 1) (test_overwrite
     rate 51.7%).

   Each rollout is graded with `countdown_all_components`. Character offsets of
   `</think>`, the `"test.py"` key and the first edited test line are saved
   together with per-token character ends, so token positions can be aligned
   offline.
3. **`capture`: response tokens.** Every rollout is replayed teacher-forced
   through the SFT base and through the checkpoint. For each response token,
   the script saves the projection onto the unit hackonly v_L at every layer and
   the full layer-20 activation (fp16, kept out of the repo).
4. **`prompts`: prompt tokens.** Each unique prompt gets a prompt-only forward.
   Attention is causal, so this gives exactly the prompt positions' activations.
5. **`analyze`: the natural direction.** The natural direction is the
   difference of rollout-mean h₂₀ between hacks and clean rollouts. It is
   cross-fitted over question parity: tokens of one half are projected on the
   direction built from the other half. Its cosine with PPS is 0.79, and the two
   halves agree at 0.998.

Classes in all plots are the no-intervention policy's own rollouts:
- **hack**: `test_overwrite`, 529 rollouts;
- **clean**: `test.py` left unchanged, solved or not, 491 rollouts.

Projections are onto the *unit* direction, so the PPS shift is a fixed +46.2 on
every axis that shows the PPS direction.

## The plots

`plot.py` writes `results/plots/<model>/`. `loreft_cosine.py` writes
`results/plots/loreft_cosine.*`. Every figure's numbers are in `numbers.json`
and `loreft_cosine.json`.

### `distributions.png`: tokens overlap, rollout means separate

Histograms of hack and clean projections on two directions, each in two
columns:
- **directions:** the PPS direction (top row) and the cross-fitted natural
  direction (bottom row);
- **columns:** per token (left) and per rollout mean (right).

The arrow is the PPS shift, starting from the clean mean. Along the natural
direction the arrow shows only PPS's component on it, 46.2 × 0.79.

| | per token | per rollout mean |
|---|---|---|
| PPS direction | d′ 1.41, AUROC 0.82 | d′ 9.75, AUROC 1.000 |
| natural direction | d′ 1.46, AUROC 0.83 | d′ 8.86, AUROC 1.000 |

The two classes differ by about 22 on average (hack +7.1, clean −15.1).
Within a class, tokens spread with a standard deviation of 14–18, while rollout
means spread by only 2–3. The shift is about 2× the class gap.

### `aligned.png`: the signal is local in time

Mean projection with a 25–75% band, against token offset, in two panels:

- **Left: aligned at the first edited `test.py` line.** For clean rollouts the
  anchor is the same line of their unedited file. In every hack the first
  edited line is 19 tokens after the `"test.py"` key.
- **Right: aligned at `</think>`.**

Both classes sit near 0 until about 70 tokens before `</think>`. Then the hacks
rise to about +27, which is where they write "we'll patch the test suite to
always return True". Clean rollouts rise less (about +15) just before
`</think>`, then drop to about −32 while copying the original `test.py`. The
arrow is the PPS shift from the early shared level. Even the peak of natural
hacking stays below what PPS adds to every token.

### `traces.png`: single rollouts

These are the raw per-token traces of one hack and one clean rollout of the same
question, with the thinking / `solution.py` / `test.py` segments shaded. The
dashed line is each rollout's mean and the dotted line is that mean plus the
PPS shift. Token-to-token swings (standard deviation 14–17) are as large as the
whole class gap.

### `segments.png`: where the per-token gap comes from

Box plots by response segment. The box is the 25–75% range and the whiskers
are 5–95%. Per-token d′:

| segment | d′ |
|---|---|
| thinking | **0.45** |
| `solution.py` | 1.33 |
| `test.py` | 3.54 |

The overall per-token d′ of 1.4 is inflated by clean rollouts copying the
original `test.py`, which sits at about −32 and makes up roughly half of all
clean tokens. In the thinking, where the decision to hack is made, a single
token barely tells the classes apart.

### `layers.png`: separation by layer

This plots d′ between hack and clean along each layer's own `hackonly` v_L, for
single tokens and for rollout means. Rollout means separate almost perfectly
from layer 7 onward, peaking at layers 20–22 (d′ ≈ 9.9). Token-level d′ stays at
or below 1.5 at every layer. So the choice of layer 20 is reasonable: the
limit is per-token, not where the vector is applied.

### `prompt.png`: PPS steers the prompt too

This figure has three panels:

- **Top.** The training sequence at layer 20: the prompt, which is the same for
  both classes, and then the response, aligned at the response start. The
  `<|im_start|>` attention-sink token at position 0 is left out because it is an
  outlier.
- **Bottom left.** The extraction prompts with the hack instruction, the
  anti-hack instruction, and the neutral eval prompts, aligned at the prompt
  end.
- **Bottom right.** Per-token histograms of prompt and response tokens, with
  the shift.

Findings:
- The prompt is about 60% of a training sequence: a median of 500 prompt tokens
  against 326 response tokens.
- Prompt tokens average −2.7 with standard deviation 10.5 (5–95%: −19 to +13).
  PPS moves all of them to around +43, above nearly every natural token.
- Hack-instruction and anti-hack-instruction prompts give the same curves
  (−1.76 vs −1.83). The direction does not respond to the instruction text.
- The most negative stretch of the prompt (mean −13.8) is the original
  `verify_solution` source embedded in the user message. This is the same text
  clean responses copy at −32. Part of what this direction measures is
  "verifier code present, unchanged", not the decision to hack.
- Hack and clean responses are indistinguishable for their first ~100 tokens.

### `loreft_cosine.png`: PPS vs the LoReFT forget directions

This is a 9×9 cosine matrix between the unit PPS v₂₀ and the write directions
of the 8 κ_F = 8 layer-20 LoReFT forget adapters, `cdhf100_gr_m64_lf1_L20_k8_s{1,2,3,4,5,9,15,16}`
(MLP retain, rank-1 LoReFT forget on layer 20's output, the same point PPS
adds to).

How the directions are defined:
- A rank-1 LoReFT edits h → h + c(h)·R, with c(h) = (W − R)·h + b. The write
  direction is R, the normalized `rot_forget`.
- R's sign is arbitrary. Each R is pointed the way its average edit pushes, using
  the natural rollout means. c is linear, so a rollout's mean coefficient is c
  of its mean h₂₀.
- Those means come from the SFT-base replay, which approximates the LoReFT
  model's own layer-20 input, since the model has retain adapters at every
  layer. This affects only the coefficient sizes, not the cosines.

Findings:
- **PPS vs LoReFT:** cosines from −0.03 to +0.02, indistinguishable from
  random. Random unit pairs in 4096 dimensions have a cosine standard
  deviation of 0.016. The LoReFTs are also nearly orthogonal to the natural hack
  direction (−0.03 to −0.07).
- **LoReFT vs LoReFT:** cosines of 0.07–0.19. The seeds share a weak learned
  component (the first principal component explains 25% of the variance), and
  that component is orthogonal to PPS as well (−0.025).
- **Edit sizes:** each adapter's edit coefficient is positive on every rollout
  and about 1.7× larger on hacks than on clean rollouts (4–17 on hacks). So the
  LoReFT edit is always on but depends on context, unlike the constant +46 of
  PPS.

## Takeaway

At layer 20, hacking shows up mostly as a slow per-rollout offset and a short,
late bump in the thinking (about +27, about 70 tokens before `</think>`). This
sits on top of token-to-token noise as large as the bump itself. PPS adds a
constant +46 at every position, about 60% of which are prompt tokens with no
hack signal. That shift is about 2× the natural hack–clean gap and overshoots
even the peak of natural hacking. Part of the direction's apparent separation
comes from verifier-code content. The GRAFT forget adapters write in an
unrelated, input-dependent direction.

Caveat: all of this uses one no-intervention checkpoint (seed 9) to supply the
natural classes. The SFT base hacked too rarely (24 test_overwrite rollouts) to
give a direction of its own.

## Reproducing

```bash
# GPU machine, checkout at $GRAFT_ROOT/GRAFT (the rl/scripts/pod layout), after
# rl/scripts/pod/setup.sh, from the checkout root:
export GRAFT_ROOT=<dir holding the GRAFT checkout>
bash experiments/steering_variance/run_pod.sh                     # full pipeline -> rl/output/steering_variance/
SV_MODE=prompts bash experiments/steering_variance/run_pod.sh     # prompt-token replay on top of it
# Copy rl/output/steering_variance/ to experiments/steering_variance/results/data/
# (acts_L20.npy files are not needed for the plots), then:
uv run --no-project --with matplotlib --with numpy python plot.py --model base   # and --model dn
# The 8 LoReFT checkpoints go to results/data/loreft/<run>/model.safetensors, then:
uv run --no-project --with torch --with safetensors --with numpy --with matplotlib \
    --index https://download.pytorch.org/whl/cpu python loreft_cosine.py
```

`run_pod.sh` downloads the SFT base and the no-intervention adapter from
Hugging Face and checks their sha256. The full layer-20 activations (fp16, about
6.6 GB per replay model) are written to `capture/<model>/acts_L20.npy` and are
not part of the repo. `results/data/` is gitignored.
