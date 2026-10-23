# GRAFT: Gradient Routing Adapter Fine Tuning

## About this fork

This is a fork of [jnward/GRAFT](https://github.com/jnward/GRAFT), the paper's code. Everything
from the paper description below onward is upstream's, apart from a few lines in the layout. This
branch (`loreft`) adds, all on the Countdown-Code setting (Qwen3-8B):

- **A reproduction of the Countdown experiments** ([`rl/REPRO.md`](rl/REPRO.md), also on branch
  `repro`): a pod re-run of Fig 2's GRAFT and No-intervention arms, a No-intervention
  anchor × learning-rate 2×2, and GRAFT / GRAFT w/o routing at lr 5e-4.
- **GRAFT with LoReFT adapters** ([`rl/LOREFT.md`](rl/LOREFT.md)): composite adapters
  (`--adapter_type composite`, MLP or LoReFT per side, on chosen layers), LoReFT on both sides, an
  MLP RETAIN adapter with a rank-1 LoReFT FORGET adapter on a few layers, and runs without the
  LeetCode anchor. Headline: a FORGET adapter of 8,193 parameters on layer 20 (κ_F 8) deploys
  with hack rate 0.013 ± 0.013 and pass@1 0.833 ± 0.003 over 8 seeds, against 0.786 ± 0.007 for
  our MLP GRAFT runs, and it holds up without the anchor (4 seeds).
- **Analyses** in [`experiments/`](experiments/): layer-window ablations of trained adapters
  (`loreft_layer_ablation/`) and the paper's PPS steering vector against natural activations
  (`steering_variance/`).
- **Data and weights:** per-run eval data, run index and checksums in
  [`rl/repro_data/`](rl/repro_data/); adapters on Hugging Face
  ([`graft-repro-adapters`](https://huggingface.co/samuel500/graft-repro-adapters),
  [`graft-loreft-adapters`](https://huggingface.co/samuel500/graft-loreft-adapters)) next to the
  SFT base ([`graft-qwen3-8b-countdown-sft`](https://huggingface.co/samuel500/graft-qwen3-8b-countdown-sft)).
- **Code:** the composite / LoReFT adapters in `rl/gradient_routing.py` and
  `rl/vllm_mlp_adapter.py` with their trainer flags, our sweep files under `rl/sweeps/`, pod
  scripts under `rl/scripts/pod/`, and `rl/tools/` helpers (HF upload, summary tables, comparison
  with the paper's released numbers).

## The paper

Code and figure data for **"Gradient Routing Localizes and Removes Unintended Behaviors in RL"**
(Ward\*, Hu\*, Wong, Hu, Woodworth, Turner, Cloud; NeurIPS 2026). \*Equal contribution.

Paper: https://arxiv.org/abs/2610.17062

GRAFT adds two MLP adapters (RETAIN and FORGET) to a frozen base model. On training samples that a
monitor flags, gradients go only to the FORGET adapter. Unflagged samples update both adapters.
A concurrent anchor environment trains the RETAIN-only policy. At deployment the FORGET adapter
is ablated. The monitor decides which parameters a target-environment sample updates; it does not
change the reward of target samples. (In the toy environments the anchor is the target environment
with a monitor-based reward penalty; see [Implementation notes](#implementation-notes).)

This repository has two independent subprojects:

| Directory | Paper sections | Contents |
|---|---|---|
| [`rl/`](rl/) | §3 Method, §4 (Countdown-Code with Qwen3-8B; six toy environments with SmolLM2-135M-Instruct), RL appendices | GRPO trainer with GRAFT, environments, rewards and monitors, sweep files, post-hoc evals, figure scripts, figure inputs |
| [`sft/`](sft/) | §5 (Qwen3-32B SFT on Terminal Wrench), SFT appendices | GRAFT SFT trainer, data prep, agentic eval pipeline (harbor + LLM judge), Fig. 4 script, eval records |

Each subproject has its own README with installation, training and evaluation instructions:
[`rl/README.md`](rl/README.md) and [`sft/README.md`](sft/README.md). Run `rl/` code from `rl/`
and `sft/` code from `sft/`; neither imports the other.

## Repository layout

```
GRAFT/
├── README.md  LICENSE  NOTICE  LICENSES/Apache-2.0.txt
├── experiments/              this fork's analyses: layer-window ablations, steering variance
├── rl/                       run everything from here
│   ├── train.py              GRPO trainer with GRAFT and the baselines
│   ├── advantages.py         routed GRPO advantages, routing masks
│   ├── gradient_routing.py   dual MLP adapters, gradient masking, adapter scales
│   ├── split_moment.py       optimizer of Eq. 1 (SplitMomentAdamW)
│   ├── trl_overrides.py      TRL 0.29 subclass (old log-probs, KL reference)
│   ├── rewards.py  rh_detectors.py  experiment_config.py
│   ├── sweep.py              sweep launcher (local GPUs with CUDA MPS, or Modal)
│   ├── vllm_*.py             vLLM generation server with in-engine dual adapters
│   ├── envs/                 toy envs, Countdown-Code, LeetCode anchor
│   ├── data/                 object_qa / cities_qa data (jsonl)
│   ├── configs/              environment / reward / monitor YAMLs
│   ├── sweeps/               one file per paper sweep
│   ├── tools/                Modal launcher, post-hoc evals, SFT priming, generalization eval
│   ├── figures/              paper figure scripts
│   ├── paper_data/           figure-input bundle (tar.gz) and per-run run_config.yaml files
│   ├── repro_data/           our Countdown runs' eval data and run index (see REPRO.md, LOREFT.md)
│   ├── repro_figures/        our Countdown figures (PNG, embedded in REPRO.md and LOREFT.md)
│   ├── REPRO.md              Countdown reproductions and follow-up experiments
│   ├── LOREFT.md             GRAFT with LoReFT adapters; runs without the LeetCode anchor
│   ├── vllm_patches/         patches for vLLM 0.17.0 (required)
│   ├── tests/                unit and equivalence tests
│   └── pyproject.toml  uv.lock  DEPENDENCIES.md
└── sft/                      run everything from here
    ├── scripts/gr/           GRAFT SFT trainer, adapters, data loader, optimizer
    ├── scripts/eval/         merge, serve, harbor eval, LLM judge, sweep drivers
    ├── scripts/              data prep, task split, GRAFT launch script
    ├── charts/plot_scatter.py   Fig. 4
    ├── build/jobs/           56 eval jobs (config, result, judge scores)
    ├── data/                 train/eval task split
    ├── docs/EVAL_SETUP.md
    ├── .env.example          API keys for the judge / HF download (copy to .env)
    └── third_party/          git submodules: harbor, terminal-wrench
```

## Quick start: regenerate figures from released data

The figure scripts need only numpy, scipy, matplotlib and pyyaml.

```bash
git clone https://github.com/jnward/GRAFT.git   # submodules are only needed for the SFT eval (sft/README.md)
cd GRAFT/rl
pip install numpy scipy matplotlib pyyaml
tar -xzf paper_data/paper_fig_inputs.tar.gz -C paper_data   # -> paper_data/{output,respmon-results}/

python figures/countdown_hf100_hf50_retain.py        # fig:countdown-hf50
python figures/countdown_generalization_scatter.py   # fig:generalization-scatter
python figures/sort_pathological_pareto.py           # fig:sort-pathological
python figures/anchor_form_pareto6.py                # fig:anchor-form
python figures/respmon6_training_curves.py           # fig:training-curves
python figures/respmon6_recall_scatter.py            # fig:recall-sweep
python figures/respmon6_training_curves_recall.py    # fig:recall-curves-{50,25,10}
python figures/hf100_arms_scatter_v2.py              # fig:dev-selection-stars

cd ../sft && python charts/plot_scatter.py           # fig:large-scale-sft (Fig. 4)
```

RL figures are written to `rl/figures/figs/` (some scripts also write `rl/final_figures/`); Fig. 4
is written to `sft/charts/`. All three locations are gitignored. The figure scripts read
`rl/paper_data/output` and `rl/paper_data/respmon-results` by default; set `GRAFT_OUTPUT_ROOT`
and `GRAFT_RESPMON_ROOT` to plot your own runs instead.
See [`rl/figures/README.md`](rl/figures/README.md) and
[`rl/paper_data/README.md`](rl/paper_data/README.md) for per-figure inputs and provenance.

## Paper → code map

Items are listed by LaTeX label, since appendix numbering may change. Paths are given from the
repository root (run `rl/` code from `rl/` and `sft/` code from `sft/`).

### Main text

| Paper item | Code / data |
|---|---|
| `fig:method` | none (diagram) |
| §3, `alg:grrl`, `eq:opt` | `rl/advantages.py`, `rl/train.py`, `rl/gradient_routing.py`, `rl/split_moment.py`; SFT: `sft/scripts/gr/` (see [Method → code](#method--code)). Unit tests: `rl/tests/`, `sft/scripts/gr/test_gr_core.py` |
| `fig:countdown-fig1` (Fig. 2), §4.1 | Figure: `rl/figures/countdown_figure1.py`. Env/reward/monitor: `rl/envs/countdown_code.py`, `rl/rewards.py`, `rl/configs/countdown_code_hack.yaml`; anchor: `rl/envs/leetcode_verified.py`, `rl/configs/leetcode_verified_anchor.yaml`. Sweeps: [`rl/sweeps/README.md`](rl/sweeps/README.md). Post-hoc evals: `rl/tools/modal_train_gr.py::eval_forget_scales` or `rl/tools/local_fseval.py`; `rl/tools/crossenv_countdown_fseval.py`; `rl/tools/expr_only_probe.py`. Baseline selection: `rl/tools/modal_ip_prescreen.py` (inoculation-prompt prescreen), `rl/tools/modal_pps_extract.py` (steering-vector layer / α), `BEST_RP` in the figure script (reward-penalty config) |
| §4.1 SFT-priming footnote | `rl/tools/countdown_build_sft_data.py`, `rl/tools/modal_countdown_sft.py`, `rl/tools/modal_countdown_hack_baserate.py`, `rl/tools/countdown_hack_baserate_grade.py` |
| `fig:objqa-schematic` | `rl/configs/object_qa_respmon.yaml`, `rl/envs/qa.py`, `rl/rewards.py`, `rl/rh_detectors.py` |
| `fig:narrative-hook` (Fig. 3), §4.2 | Figure: `rl/figures/proto_respmon6_rv2canon_sidebyside.py`. Configs: see [Toy environment configs](#toy-environment-configs); sweeps in `rl/sweeps/` |
| `fig:large-scale-sft` (Fig. 4), §5 | Figure: `sft/charts/plot_scatter.py`. Training: `sft/scripts/launch_gr_32b_split.sh` → `sft/scripts/gr/train_gr.py`. Data: `sft/scripts/data_prep.py` → `sft/scripts/split_train_eval.py` → `sft/scripts/gr/data_prep_gr.py`. Eval: `sft/scripts/eval/` |
| `sec:discussion` (§7), gradient-norm separability | `rl/train.py --adapter_diag_level per_sample_recompute`; `rl/tools/gen_grad_diag_html.py`, `rl/tools/gen_separability_html.py`, `rl/tools/analyze_grad_diag_*.py` |

### Appendix

| Paper item | Code / data |
|---|---|
| `app:algorithm-details` (GRPO loss, π_old, amplification, baselines) | `rl/trl_overrides.py`; `rl/train.py` (`_packed_compute_loss`, IS-path selection); `rl/advantages.py` (`routing_grad_mask_weights`, `_baseline_nonflagged_var_all`); see the π_old note under [Implementation notes](#implementation-notes) |
| `app:countdown-prompt` | `rl/envs/countdown_code.py` (`build_prompt`) |
| `tab:countdown-generalization`, `fig:generalization-scatter`, `tab:generalization-perturbations`, `app:generalization-full` | `rl/tools/countdown-generalization-eval/` (checkpoint specs, scaffolds, graders, Modal generation, `make_table.py`, `make_appendix_table.py`); figure `rl/figures/countdown_generalization_scatter.py` (error bars are SEM). Checkpoint sets: `--spec countdown_hf100_all` (two-file tables and scatter), `--spec countdown_hf100_gr` (perturbation scaffolds); base model: `modal_humaneval_generate.py::run --model /output/countdown_sft_model/qwen3-8b --label sft-base` |
| `tab:countdown-noanchor` | `rl/sweeps/countdown_hf100_gr_nocoh_lr3.py`, `rl/tools/local_fseval.py`, `rl/tools/collate_countdown_fseval.py` |
| `fig:countdown-hf50` | `rl/figures/countdown_hf100_hf50_retain.py`; `rl/sweeps/countdown_hf50_*.py` |
| `fig:sort-pathological` | `rl/figures/sort_pathological_pareto.py`; `rl/configs/sorting_copy_respmon.yaml` |
| `fig:anchor-form` | `rl/figures/anchor_form_pareto6.py`; anchor-form sweeps in `rl/sweeps/` |
| `fig:training-curves` | `rl/figures/respmon6_training_curves.py` |
| `fig:recall-sweep`, `fig:recall-curves-{50,25,10}` | `rl/figures/respmon6_recall_scatter.py`, `rl/figures/respmon6_training_curves_recall.py`; `rl/sweeps/respmon_recall_suite.py` |
| `fig:dev-selection-stars` | `rl/figures/hf100_arms_scatter_v2.py` |
| `app:ip-prompts` | Countdown: `IP_PROMPTS` in `rl/sweeps/countdown_hf50_ip.py` (mirrored in `rl/tools/modal_ip_prescreen.py`) |
| `app:envs`, `tab:env-mechanics-A`, `tab:env-prompts` | `rl/envs/{addition,qa,qa_utils,repeat,sorting}.py`, `rl/rewards.py`, `rl/rh_detectors.py`, `rl/experiment_config.py`, `rl/data/*.jsonl`; configs: see [Toy environment configs](#toy-environment-configs) |
| `app:small-envs` background paragraph (swapped detectable/undetectable sets) | `rl/configs/*_inv.yaml`, `rl/configs/*_inverted.yaml`; their runs in `rl/sweeps/respmon_rv2_inverted_gr_3seed.py` and `rl/sweeps/respmon_ivainv_dn_rp_3seed.py` |
| `app:hyperparams`, SmolLM2 | `rl/sweeps/respmon_repro_coh128_lam1_3seed.py` (`_shared`, `_new`), `rl/sweeps/_respmon_canon.py`; `rl/paper_data/run_configs/respmon-results/` |
| `app:hyperparams`, Qwen3-8B | `rl/sweeps/countdown_code_rp.py` (`_base`), `rl/sweeps/countdown_code_gr.py` (`_gr`), `rl/sweeps/countdown_hf100_gr_lccoh64_lr3.py`; `rl/paper_data/run_configs/output/` |
| `app:hyperparams`, adapter architecture / MLP vs LoRA | `rl/gradient_routing.py` (`DualMLPAdapter`, `apply_dual_mlp`; LoRA: `DualLoRALinear`, `apply_dual_lora`), `rl/vllm_utils.py` (`MLP_PRESETS`), `rl/vllm_lora.py`, `sft/scripts/gr/adapter.py`; see the LoRA note under [Implementation notes](#implementation-notes) |
| `app:hyperparams`, software stack | `rl/train.py`, `rl/trl_overrides.py`, `rl/vllm_*.py`, `rl/vllm_patches/` |
| `app:hyperparams`, large-scale SFT and SFT evaluation; `app:hack-prompt` | `sft/scripts/launch_gr_32b_split.sh`, `sft/scripts/gr/`, `sft/scripts/eval/`; elicitation prompt `prompts/hack_prompt_v5.md` in the Terminal Wrench submodule; see [`sft/README.md`](sft/README.md) |
| `app:compute` | none |
| Code availability | this repository |

### Toy environment configs

| Paper name | Config | `environment` |
|---|---|---|
| repeat_extra | `rl/configs/repeat_respmon.yaml` | `repeat` |
| addition | `rl/configs/addition_v2_syco_respmon_iva.yaml` | `addition_v2` |
| sort | `rl/configs/sorting_copy_framed.yaml` | `sorting_framed` |
| object_qa | `rl/configs/object_qa_respmon.yaml` | `object_qa` |
| cities_qa | `rl/configs/cities_qa_syco_respmon_iva.yaml` | `cities_qa` |
| persona_qa | `rl/configs/persona_qa_flattery_respmon_bvo.yaml` | `persona_qa` |

- Other toy configs: `sorting_copy_respmon.yaml` is the stock sort of `fig:sort-pathological`;
  `object_qa_syco_respmon_iva.yaml` is object_qa with the monitored and unmonitored words swapped
  (monitor = indeed, as in the addition and cities_qa configs; used in `respmon_rv2_gr_3seed`,
  `respmon_iva_{dn,rp}_3seed` and `respmon_rv2_vercoh128_gr_3seed`); `sorting_copy_framed_v99.yaml`
  is a framed-sort pilot variant (`sort_framed_pilot.py`); the `*_inv.yaml` / `*_inverted.yaml`
  configs (addition, cities, persona, repeat, stock sort) swap the monitored and unmonitored forms
  (`app:small-envs` background paragraph).
- Runs of `object_qa_respmon.yaml` are named `object_qa_syco_respmon_iva_inv_*` after its `name`
  field (run-name conventions: [`rl/sweeps/README.md`](rl/sweeps/README.md)). Their `run_config.yaml`
  files in `rl/paper_data/` record it as `config_path: configs/object_qa_syco_respmon_iva_inv.yaml`.
- `rl/sweeps/_respmon_canon.py:CANON_ENVS` (per-env steps) lists the stock sort config. The
  framed-sort arms come from the `sort_framed_*`, `respmon_sortframed_*` and `respmon_recall_suite`
  sweeps and from the framed-sort override in `respmon_noroute_canon_3seed.py`.

## Method → code

| Algorithm 1 / Eq. 1 | RL (`rl/`) | SFT (`sft/scripts/gr/`) |
|---|---|---|
| RETAIN / FORGET adapters (parallel SwiGLU branch in every MLP) | `gradient_routing.DualMLPAdapter`, `apply_dual_mlp`; widths in `vllm_utils.MLP_PRESETS` (`m16` SmolLM2, `m64` Qwen3-8B) | `adapter.DualMLPAdapter`, `inject_adapters` |
| π_{R+F} / π_R (adapter enable) | `gradient_routing.set_scales(model, retain, forget)`; eval modes `both` / `retain_only` / `forget_only` | `adapter.set_scales` |
| Target and anchor rollouts | vLLM generation in `train.py`; anchor slots via `--coh_samples_per_rollout`, `--coh_config` (cross-env anchor, e.g. LeetCode for Countdown) and `--coh_prompt_source` | anchors = 1% of unlabeled data (`--classifier-retain-recall 0.01 --retain-from-unlabeled`; `loader.py`) |
| Monitor M | `rh_detector` in the config YAML → `experiment_config.build_rh_detector` → `rh_detectors.py` | simulated monitor in data prep: `data_prep_gr.py --recall` labels that fraction of ground-truth hacks as flagged |
| Advantages (baseline over unflagged samples, std over the whole group) | `advantages.compute_routed_advantages`, `_baseline_nonflagged_var_all` | n/a (cross-entropy loss) |
| Gradient masks, as (forward FORGET scale, RETAIN grad weight, FORGET grad weight): anchor (0,1,0), unflagged (1,1,1), flagged (1,0,2) | `advantages.routing_grad_mask_weights`, `adapter_kappas`; applied in one packed forward/backward in `train.SampleGRPOTrainer._fused_forward_backward` via the stop-gradient decomposition `gradient_routing._fused_decouple` | `train_gr.py` (`--kappa 2`) |
| Optimizer, Eq. 1 (first moment from routed gradient, second moment from unrouted gradient) | `split_moment.SplitMomentAdamW`; unrouted gradient captured by `gradient_routing.PreRoutingGradAccumulator` | `split_moment.SplitMomentAdamW` (`--split-moment`) |
| Deploy π_R | eval mode `retain_only`; post-hoc forget-scale eval at scale 0.0 | `--mode retain_only` in `scripts/eval/run.py` |

Baselines in `rl/`: no intervention (`routing_mode: none`), reward penalty
(`reward_penalty_baseline`, `reward_penalty_amount`), filtering (`filter_baseline`), GRAFT w/o
routing (`rh_detector_recall: 0.0`), GRAFT w/o anchoring (no anchor samples), inoculation
prompting (`countdown_train_system_suffix`) and preventative steering (`pps_vector_path`,
`pps_layer`, `pps_alpha`; vector from `rl/tools/modal_pps_extract.py`).

## Implementation notes

These describe what the code does. Where details matter for reproduction they are expanded in
`rl/README.md` and `sft/README.md`.

- **π_old and clipping.** Toy runs take one optimizer step per rollout with
  `vllm_importance_sampling: false`, so the old log-probs are not computed and the PPO ratio is
  identically 1 (the clip never binds). Framed sort (`sorting_framed`) is not in `train.py`'s
  `_SMALL_SCALE_ENVS` and takes the slow path with trainer-recomputed old log-probs. Countdown runs
  take 4 optimizer steps per rollout with recomputed old log-probs and truncated vLLM importance
  sampling.
- **Optimizer details.** `SplitMomentAdamW` as used in training also has a per-coordinate step
  bound (`--graft_w_max`, default 4.0) and applies the gradient-clipping coefficient to the
  unrouted gradient too. It freezes the FORGET adapter (no m/v/step-count/weight-decay update) on
  any optimizer window in which no target-environment sample is trained on, i.e. every sample
  that reaches the backward pass is an anchor sample (`n_routing == 0` in
  `train.SampleGRPOTrainer._fused_forward_backward`). This includes windows whose target samples
  were all removed before the backward pass (by `--drop_zero_advantage`, or unflagged samples
  during `--forget_warmup_steps` when none are flagged).
- **Monitor scope.** With `detect_unhackable: true` (the default) the monitor also flags, routes and
  penalizes hack forms on unhackable prompts. The addition, cities_qa, object_qa and persona_qa
  monitors are case-insensitive substring tests; repeat_extra's `repeat_contiguous` flags only
  contiguous repeats, and sort's `sorting_copy_verbatim` requires a positional copy fraction above
  0.5 plus a verbatim match of the input sequence (`rl/rh_detectors.py`).
- **Toy GRAFT w/o routing** is GRAFT with `rh_detector_recall: 0`. In toy envs the anchor slice is
  penalized using the monitor, so this setting also removes that penalty; lower recall likewise
  thins it.
- **Countdown reward-penalty arm** also trains on the LeetCode anchor (64 samples per rollout;
  `_ANCHOR` in `rl/sweeps/countdown_hf50_rp_lccoh64.py`).
- **Toy evaluation.** 256 fixed eval prompts per adapter mode (seed 99, test split), one sample at
  temperature 1, every 10 steps. Figures average the last 10% of eval rows. The "Base model" point
  is the first eval row, i.e. step 10 (step 0 only for the `sort_framed_*` runs, which set
  `eval_at_start`). All other toy run configs set `eval_at_start: false`, while this code
  defaults to true; pass `--no-eval_at_start` when re-running them (`rl/README.md`).
- **Toy model** is `HuggingFaceTB/SmolLM2-135M-Instruct`.
- **Run configs.** `run_config.yaml` omits argparse-only arguments (e.g. `max_grad_norm`); the
  sweep files, with the configs they name, are authoritative. The framed-sort `run_config.yaml`
  files in `rl/paper_data/` have no `sort_prompt_style` key; those runs used `framing_noverb`, which
  `rl/configs/sorting_copy_framed.yaml` sets.
- **Logged detector channels.** The `hack_freq_{,un}detectable/<detector>` channels of the toy and
  Countdown configs are all-prompt rates, not hackable-prompt quadrants.
- **LoRA adapters and vLLM.** The vLLM LoRA path ignores adapter scales (`vllm_lora.py`'s
  `set_scales` is a no-op), so in-training retain-only / forget-only evals of LoRA runs are not
  ablated; measure LoRA localization with an HF-side eval that applies
  `gradient_routing.set_scales`.
- **`sweep.py`** prints `[WARN] Failed to regenerate sweep pages: …` about once a minute and
  `[WARN] Failed to generate plots for group …` (optional dashboard hooks); these are expected.
- **`python train.py` needs `--config configs/<x>.yaml`** (or an explicit `--environment`). Sweeps
  always pass a config.
- **SFT.** GRAFT and the SFT baselines use different optimizer settings; `--match-rslora` makes
  `--lora-r` inert. See `sft/README.md`.

## Running experiments

- **RL:** install with `uv sync` in `rl/` (Python 3.11, vLLM 0.17.0, flash-attn wheels for Linux
  x86_64 / CUDA 12.8 / torch 2.10), then `source .venv/bin/activate && bash vllm_patches/apply.sh`
  (the script patches the vLLM of the `python` on `PATH`). Toy sweeps ran locally under
  CUDA MPS; most Countdown sweeps ran on Modal (a few on a local GPU pod):

  ```bash
  cd rl
  python -u sweep.py --name respmon_rv2_gr_3seed --config sweeps/respmon_rv2_gr_3seed.py --no_baseline
  python sweep.py --name countdown_hf100_gr_lccoh64_lr3 \
      --config sweeps/countdown_hf100_gr_lccoh64_lr3.py --backend modal --no_pack --no_baseline
  ```

  Countdown and the LeetCode anchor need a clone of
  [ariahw/rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking) at `RH_REPO_PATH`, an
  SFT-primed Qwen3-8B (`GRAFT_COUNTDOWN_SFT_MODEL`) and, for preventative steering, a steering
  vector (`GRAFT_PPS_VECTOR`). Details, Modal setup and
  the full sweep → paper table:
  [`rl/README.md`](rl/README.md), [`rl/sweeps/README.md`](rl/sweeps/README.md).
- **SFT:** from the repo root `git submodule update --init`; then
  `cd sft && cp .env.example .env && bash scripts/eval/setup_box.sh` on the eval host; prepare the
  training data as in [`sft/README.md`](sft/README.md) ("Data pipeline"); train GRAFT with
  `GRAFT_SFT_VENV=.venvs/eval GRAFT_SFT_DATA_DIR=<data dir> GRAFT_SFT_CKPT_DIR=<ckpt dir> bash scripts/launch_gr_32b_split.sh`
  (the launcher's defaults are `.venv` and `/workspace/training/{data,checkpoints/gr}`). Baselines,
  evaluation and the job → arm table: [`sft/README.md`](sft/README.md),
  [`sft/docs/EVAL_SETUP.md`](sft/docs/EVAL_SETUP.md).

Many RL tests need a CUDA GPU, vLLM 0.17 with the patches applied, or model downloads.

## Data

- `rl/paper_data/paper_fig_inputs.tar.gz`: metrics-only inputs of the RL figures
  (`routing_eval.jsonl`, `run_config.yaml`, post-hoc eval JSONs, probe JSONs, generalization
  `graded.jsonl`). File list and checksums in `rl/paper_data/MANIFEST.txt`.
- `rl/paper_data/run_configs/`: `run_config.yaml` for 37 sweep dirs (371 files). Further
  `run_config.yaml` files are inside the tarball next to their logs (for `respmon_recall_suite`,
  `respmon_canon_rp_mix_3seed`, `respmon_canon_filter_3seed`, `respmon_sortframed_*` and
  `respmon_canon_gr_seeds4to8`, only there).
- `sft/build/jobs/`: 56 SFT eval jobs (`config.json`, `result.json`,
  `judge_scores_judge_v3.json`): the 45 jobs plotted in Fig. 4, 4 GA×1/×2 jobs, and 7 additional
  eval jobs (`gr-s1like-unc-*`, `gr-s1like-inoc-v5-*`), each shipped with its config; the job →
  arm table is in [`sft/README.md`](sft/README.md). The judge files include the judge model's
  free-text reasoning, which quotes rollout commands and task content.

The SFT-primed Qwen3-8B is built with `rl/tools/countdown_build_sft_data.py` and
`rl/tools/modal_countdown_sft.py`, and the steering vector with `rl/tools/modal_pps_extract.py`
(see `rl/README.md`). Third-party datasets and models are downloaded at runtime (see `NOTICE`).

## Safety

The code graders execute model-written Python under resource limits only, not in a sandbox. Run
training and evaluation inside a container or VM. The SFT eval serves models with vLLM on
`0.0.0.0:8000` by default without authentication; on a shared host, firewall the port or set
`VLLM_HOST=127.0.0.1`.

## External dependencies and licenses

The repository is MIT-licensed (`LICENSE`). Some files are modified copies of, or contain text
from, third-party projects (vLLM, Hugging Face TRL, Countdown-Code, Terminal Wrench, harbor,
rl-rewardhacking); `NOTICE` lists them with their licenses, and `LICENSES/Apache-2.0.txt` has the
Apache-2.0 text. Not vendored:

- [harbor](https://github.com/harbor-framework/harbor) and
  [Terminal Wrench](https://github.com/few-sh/terminal-wrench): git submodules under
  `sft/third_party/`, pinned to harbor `290ab985` and terminal-wrench `d8a29613`.
- [rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking): cloned separately (`RH_REPO_PATH`);
  provides the code-test runner and the LeetCode data.
- [Countdown-Code](https://github.com/zohaib-khan5040/Countdown-Code): source of the o4-mini traces
  used for SFT priming.
- Models and datasets from the Hugging Face Hub, downloaded at runtime.

## Citation

```bibtex
@inproceedings{ward2026graft,
  title     = {Gradient Routing Localizes and Removes Unintended Behaviors in {RL}},
  author    = {Ward, Jake and Hu, Shawn and Wong, Aria and Hu, Nathan and Woodworth, Bryce and
               Turner, Alexander Matt and Cloud, Alex},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {arXiv link TBD}
}
```
