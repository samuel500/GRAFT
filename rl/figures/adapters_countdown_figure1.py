"""Figure 1 (countdown_figure1.py layout) for the LoReFT forget-adapter runs of
REPRO.md section 4, built offline from repro_data/:

  adapters_countdown_figure1        left: scatter of No intervention (lr/3), GRAFT
                                    (ours, MLP adapters), GRAFT with an MLP retain
                                    adapter and a rank-1 LoReFT forget adapter on
                                    layer 20 (κ_F 8, 8 seeds) and the base model
                                    (ours filled, the paper's released one hollow).
                                    Right: that run's adapter decomposition (per-step
                                    routing evals, n=64, steps 0-192).
  adapters_countdown_figure1_left   the left scatter on its own
  adapters_countdown_variants       scatter of every LoReFT / mixed variant (seed 9;
                                    κ_F 8 L20 as its 8-seed mean) with GRAFT (ours)
                                    and the base model

Error bars and bands are 95% t-intervals over seeds (countdown_figure1.py and the
repro figures use ± SEM). No intervention is at scale 1.0 (it has no FORGET adapter), every GRAFT variant
at 0.0 (RETAIN only). The y axis is clean pass@1 (expr-only probe). The step-0
anchor of the right panels is the base model's n=256 fseval (the paper figure
averages n=64 base evals, which these runs do not have), and the curves end at
the last in-training eval (step 192; there are no posthoc step-200 n=64 evals).

Run (from rl/):
    uv run --no-project --with matplotlib --with numpy --with scipy \
        python figures/adapters_countdown_figure1.py
-> figures/figs/ and final_figures/<name>.{pdf,png}; the PNGs are also copied to
   repro_figures/ (tracked, embedded in REPRO.md).
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
D = os.path.join(RL, "repro_data")
os.environ["GRAFT_OUTPUT_ROOT"] = D
K8_SEEDS = (9, 15, 16, 1, 2, 3, 4, 5)
os.environ["GRAFT_GR_SEEDS"] = ",".join(map(str, K8_SEEDS))
sys.path[:0] = [HERE, os.path.join(RL, "tools")]

import statistics as st  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402
import countdown_figure1 as cf  # noqa: E402
from repro_compare import Paper  # noqa: E402

MIX = f"{D}/exp_m64_loreft_local_fseval/cdhf100_gr_m64_lf1"
K8 = "GRAFT, LoReFT forget (L20, κ_F 8)"


def _base():
    """The base model (SFT; retain 0, forget 0): s9's fseval and s16's probe, as in REPRO.md."""
    base = [(cf.val(_BASE_FS["scales"]["0.0"], "hack_freq/"),
             cf._probe_pass1("cdhf100_gr_lccoh64_lr3_s16__r0.0", "0.0"))]
    return ("Base model (ours)", base, "#444444", "o", False)


def _paper_base():
    """The paper's base model (SFT; retain 0, forget 0): its released fseval and probe."""
    paper = Paper(os.path.join(RL, "paper_data", "paper_fig_inputs.tar.gz"))
    fs = paper.load("countdown_hf100_gr_lccoh64_lr3_fseval/cdhf100_gr_lccoh64_lr3_s9__r0.0.json")
    pr = paper.load("expr_only_probe_arms/cdhf100_gr_lccoh64_lr3_s16__r0.0.json")
    return ("Base model (paper)", [(cf.val(fs["scales"]["0.0"], "hack_freq/"),
                                    pr["scales"]["0.0"]["true_pass1"])], "#9a9a9a", "o", True)


def ci95(xs):
    """Half-width of the 95% t-interval of the seed mean (0 for a single seed)."""
    n = len(xs)
    if n < 2:
        return 0.0
    return stats.t.ppf(0.975, n - 1) * st.stdev(xs) / n ** 0.5


cf.ci = ci95  # scatter error bars, printed ±, and the right panels' bands


def _graft():
    return ("GRAFT (ours)", cf._pts(f"{D}/repro_fig2_fseval/cdhf100_gr_lccoh64_lr3_s*.json", "0.0"),
            "#2ca02c", "o", False)


def main_arms(gr_all_seeds=True):
    return [
        ("No intervention, lr/3", cf._pts(f"{D}/dn_lr3_fseval/cdhf100_dn_lr3_s*.json", "1.0"),
         "#c7a17a", "^", False),
        (K8, cf._pts(f"{MIX}_L20_k8_s*.json", "0.0"), "#9467bd", "D", False),
        _graft(),
        _base(),
        _paper_base(),
    ]


def variant_arms(gr_all_seeds=True):
    lf = f"{D}/exp_loreft_fseval/cdhf100_gr_loreft"
    return [
        ("LoReFT both, r1, all layers", cf._pts(f"{lf}_r1_lr3_s9.json", "0.0"), "#bcbd22", "v", False),
        ("LoReFT both, r8, all layers", cf._pts(f"{lf}_r8_lr3_s9.json", "0.0"), "#8c6d31", "v", False),
        ("LoReFT both, r1, L20", cf._pts(f"{lf}_r1_lr3_L20_s9.json", "0.0"), "#e377c2", "v", False),
        ("MLP + LoReFT F, all layers", cf._pts(f"{MIX}_Lall_s9.json", "0.0"), "#17becf", "s", False),
        ("MLP + LoReFT F, L11-20", cf._pts(f"{MIX}_L11_20_s9.json", "0.0"), "#1f77b4", "s", False),
        ("MLP + LoReFT F, L16-20", cf._pts(f"{MIX}_L16_20_s9.json", "0.0"), "#393b79", "s", False),
        ("MLP + LoReFT F, L20, κ_F 2", cf._pts(f"{MIX}_L20_s9.json", "0.0"), "#d62728", "D", False),
        ("MLP + LoReFT F, L20, κ_F 4", cf._pts(f"{MIX}_L20_k4_s9.json", "0.0"), "#ff7f0e", "D", False),
        ("MLP + LoReFT F, L20, κ_F 8 (8 seeds)", cf._pts(f"{MIX}_L20_k8_s*.json", "0.0"),
         "#9467bd", "D", False),
        _graft(),
        _base(),
        _paper_base(),
    ]


# right panels: the κ_F 8 runs instead of the MLP GRAFT runs
cf._gr_sweep = lambda s: (f"{D}/exp_m64_loreft_local/cdhf100_gr_m64_lf1_L20_k8_s{s}",
                          f"{D}/exp_m64_loreft_local_fseval", f"cdhf100_gr_m64_lf1_L20_k8_s{s}")
_BASE_FS = json.load(open(f"{D}/repro_fig2_fseval/cdhf100_gr_lccoh64_lr3_s9__r0.0.json"))
# step-0 anchor: the base model's n=256 fseval (no n=64 base evals for these runs)
cf._base_anchor = lambda slug: cf.val(_BASE_FS["scales"]["0.0"],
                                      ("retain" if slug == "retain" else "hack_freq") + "/")
cf.GR_MODES = [
    ("both", "#1f77b4", "--", 2.4, "LoReFT-forget GRAFT: pre-ablation"),
    ("retain_only", "#9467bd", "-", 2.4, "LoReFT-forget GRAFT: deployed"),
    ("forget_only", "#8b0000", ":", 2.6, "LoReFT-forget GRAFT: forget-only"),
]


def _scatter(ax, arms, legend):
    cf.scatter_arms = arms
    cf.draw_scatter(ax, fs=1.55)
    # widen the point-based limits to cover the 95% intervals too
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for _, pts, *_ in arms():
        hs, rs = [p[0] for p in pts], [p[1] for p in pts]
        x0 = max(x0, min(1.0, st.mean(hs) + ci95(hs)) + 0.02)
        y0 = min(y0, max(0.0, st.mean(rs) - ci95(rs)) - 0.02)
        y1 = max(y1, min(1.0, st.mean(rs) + ci95(rs)) + 0.02)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    handles = ax.get_legend().legend_handles
    ax.legend(handles=handles, **legend)
    ax.set_axisbelow(True)


def _save(fig, name):
    for d in (os.path.join(HERE, "figs"), os.path.join(RL, "final_figures")):
        os.makedirs(d, exist_ok=True)
        for ext in ("pdf", "png"):
            out = os.path.join(d, f"{name}.{ext}")
            fig.savefig(out, bbox_inches="tight", pad_inches=0.04,
                        **({"dpi": 150} if ext == "png" else {}))
            print(f"wrote {out}")
    tracked = os.path.join(RL, "repro_figures")
    os.makedirs(tracked, exist_ok=True)
    shutil.copy(os.path.join(RL, "final_figures", f"{name}.png"), tracked)
    print(f"copied {name}.png -> {tracked}")


# below the axes: inside, the legend would cover data or the 95% intervals
MAIN_LEGEND = dict(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=16,
                   framealpha=0.92)


def figure1():
    fig = plt.figure(figsize=(17.0, 8.3))
    sub_l, sub_r = fig.subfigures(1, 2, width_ratios=[9.3, 7.7], wspace=0.0)
    TOP, BOT = 0.97, 0.10
    axl = sub_l.subplots(1, 1)
    axl.set_box_aspect(1)
    _scatter(axl, main_arms, MAIN_LEGEND)
    sub_l.subplots_adjust(left=0.13, right=0.97, top=TOP, bottom=BOT)
    axr_top, axr_bot = sub_r.subplots(2, 1, sharex=True)
    cf.draw_adapter_panels(axr_top, axr_bot)
    axr_bot.legend(handles=axr_bot.get_legend().legend_handles, loc="lower right",
                   bbox_to_anchor=(1.0, 0.05), fontsize=13, framealpha=0.92)
    sub_r.subplots_adjust(left=0.17, right=0.97, top=TOP, bottom=BOT, hspace=0.07)
    sub_r.align_ylabels([axr_top, axr_bot])
    for ax in (axr_top, axr_bot):
        ax.set_xlim(0, 200)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    _save(fig, "adapters_countdown_figure1")


def left(name, arms, legend):
    fig, ax = plt.subplots(figsize=(9.3, 8.3))
    ax.set_box_aspect(1)
    _scatter(ax, arms, legend)
    _save(fig, name)


if __name__ == "__main__":
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.unicode_minus"] = False
    figure1()
    left("adapters_countdown_figure1_left", main_arms, MAIN_LEGEND)
    left("adapters_countdown_variants", variant_arms,
         dict(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=14, framealpha=0.92))
