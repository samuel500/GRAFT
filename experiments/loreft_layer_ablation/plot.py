"""Plots for the layer-window ablations, from results/<run_name>/configs/, into
results/<run_name>/plots/:

  retain_windows.png  Part A: solve rate, probe pass@1, hack rate per retain window
  forget_windows.png  Part B: the same per forget window
  effect_share.png    per window, the share of the full effect that the window
                      gives on its own (keep) and that is lost without it (drop):
                      Part A solve-rate gain, Part B hack-rate increase

Error bars are ±1 binomial SE (every fseval and probe metric is a 0/1 rate, n=256).

With --aggregate OUT and several run dirs, the same three plots go to OUT/plots/
for the mean over runs (seeds), with ±1 SEM across runs; effect shares are
computed per run, then averaged.

    uv run --no-project --with matplotlib python experiments/loreft_layer_ablation/plot.py \
        experiments/loreft_layer_ablation/results/cdhf100_gr_loreft_r1_lr3_s9
    uv run --no-project --with matplotlib python experiments/loreft_layer_ablation/plot.py \
        --aggregate results/<group> results/<run_s1> results/<run_s9> ...
"""
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference palette (dataviz skill, light mode): categorical slots 1-2, text, grid.
SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
KEEP, DROP = "#2a78d6", "#eb6834"

METRICS = [  # (key, panel title)
    ("solve", "Solve rate (fseval retain reward)"),
    ("probe", "Expr-only probe pass@1"),
    ("hack", "Hack rate (fseval)"),
]


def load(run_dir):
    rows = {}
    for p in glob.glob(os.path.join(run_dir, "configs", "*.json")):
        r = json.load(open(p))
        row = {"probe": r["probe"]["true_pass1"], "n": r["n"]}
        for short, head in (("solve", "retain"), ("hack", "hack_freq")):
            (k,) = [k for k in r["fseval"] if k.split("/")[0] == head]
            row[short] = r["fseval"][k]
        rows[r["config"]["name"]] = row
    return rows


def se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def mean_sem(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def with_se(rows):
    """Single run: each metric's error is its binomial SE."""
    for row in rows.values():
        for key, _ in METRICS:
            row[key + "_e"] = se(row[key], row["n"])
    return rows


def aggregate(rowsets):
    """Several runs: per config and metric, the mean over runs and the SEM across runs."""
    names = set.intersection(*(set(r) for r in rowsets))
    assert all(set(r) == names for r in rowsets), "runs have different configs"
    out = {}
    for name in names:
        row = {"n": rowsets[0][name]["n"]}
        for key, _ in METRICS:
            row[key], row[key + "_e"] = mean_sem([r[name][key] for r in rowsets])
        out[name] = row
    return out


def windows_of(rows):
    ws = {n.split("_", 2)[2] for n in rows if "_keep_" in n}
    return sorted(ws, key=lambda w: int(w.split("-")[0]))


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ref_line(ax, y, label, ls, color):
    ax.axhline(y, color=color, linewidth=1.2, linestyle=ls, zorder=1)
    ax.annotate(f"{label} {y:.2f}", xy=(1.0, y), xycoords=("axes fraction", "data"),
                xytext=(4, 0), textcoords="offset points", va="center", ha="left",
                fontsize=8, color=TEXT_2)


def window_figure(rows, part, refs, title, note, out):
    ws = windows_of(rows)
    x = list(range(len(ws)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for ax, (key, ptitle) in zip(axes, METRICS):
        style(ax)
        (on_name, on_label), (off_name, off_label) = refs
        ref_line(ax, rows[on_name][key], on_label, "--", TEXT_2)
        ref_line(ax, rows[off_name][key], off_label, ":", MUTED)
        for kind, color, label, dx in (("keep", KEEP, "Only this window on", -0.08),
                                       ("drop", DROP, "All but this window on", 0.08)):
            ys = [rows[f"{part}_{kind}_{w}"][key] for w in ws]
            es = [rows[f"{part}_{kind}_{w}"][key + "_e"] for w in ws]
            ax.errorbar([i + dx for i in x], ys, yerr=es, color=color, linewidth=2,
                        marker="o", markersize=7, markeredgecolor=SURFACE,
                        markeredgewidth=2, elinewidth=1, capsize=0, label=label, zorder=3)
        ax.set_xticks(x, [f"L{w}" for w in ws])
        ax.set_title(ptitle, fontsize=11, color=TEXT, loc="left")
        lo = min(min(ax.get_ylim()[0], 0), 0)
        ax.set_ylim(lo, ax.get_ylim()[1])
    axes[0].set_xlabel("Layer window (0-indexed; layer 0 never ablated)", fontsize=9, color=TEXT_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9,
               labelcolor=TEXT, bbox_to_anchor=(0.98, 1.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.text(0.01, 0.905, f"{note} Dashed and dotted lines: all-on and all-off references.",
             fontsize=9, color=TEXT_2)
    fig.tight_layout(rect=(0, 0, 0.95, 0.9))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def share(rows, part, key, on, off, w):
    full = rows[on][key] - rows[off][key]
    n = rows[on]["n"]
    keep = rows[f"{part}_keep_{w}"][key]
    drop = rows[f"{part}_drop_{w}"][key]
    s_keep = (keep - rows[off][key]) / full
    s_drop = (rows[on][key] - drop) / full
    e_keep = math.hypot(se(keep, n), se(rows[off][key], n)) / abs(full)
    e_drop = math.hypot(se(drop, n), se(rows[on][key], n)) / abs(full)
    return s_keep, e_keep, s_drop, e_drop, full


def share_runs(rowsets, part, key, on, off, w):
    """share() for one run; for several, the mean of the per-run shares ± SEM across runs."""
    if len(rowsets) == 1:
        return share(rowsets[0], part, key, on, off, w)
    per = [share(r, part, key, on, off, w) for r in rowsets]
    (k, ek), (d, ed) = mean_sem([v[0] for v in per]), mean_sem([v[2] for v in per])
    return k, ek, d, ed, sum(v[4] for v in per) / len(per)


def effect_figure(rowsets, note, out):
    ws = windows_of(rowsets[0])
    x = list(range(len(ws)))
    panels = [
        ("retain", "solve", "retain_all", "retain_none",
         "Part A, retain: solve-rate gain over the SFT model"),
        ("forget", "hack", "forget_all", "retain_all",
         "Part B, forget: hack-rate increase over forget off"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE, sharey=True)
    bw = 0.34
    for ax, (part, key, on, off, ptitle) in zip(axes, panels):
        style(ax)
        vals = [share_runs(rowsets, part, key, on, off, w) for w in ws]
        full = vals[0][4]
        ax.axhline(1.0, color=TEXT_2, linewidth=1.2, linestyle="--", zorder=1)
        ax.axhline(0.0, color=MUTED, linewidth=1.0, zorder=1)
        ax.annotate("full effect", xy=(1.0, 1.0), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                    color=TEXT_2)
        ax.bar([i - bw / 2 - 0.01 for i in x], [v[0] for v in vals], bw, yerr=[v[1] for v in vals],
               color=KEEP, error_kw={"elinewidth": 1, "ecolor": TEXT_2},
               label="Window alone (keep only this window)", zorder=2)
        ax.bar([i + bw / 2 + 0.01 for i in x], [v[2] for v in vals], bw, yerr=[v[3] for v in vals],
               color=DROP, error_kw={"elinewidth": 1, "ecolor": TEXT_2},
               label="Lost without it (drop only this window)", zorder=2)
        ax.set_xticks(x, [f"L{w}" for w in ws])
        ax.set_title(f"{ptitle} (full = {full:+.3f}{', mean' if len(rowsets) > 1 else ''})",
                     fontsize=11, color=TEXT, loc="left")
    axes[0].set_ylabel("Share of full effect", fontsize=9, color=TEXT_2)
    axes[0].set_xlabel("Layer window", fontsize=9, color=TEXT_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9,
               labelcolor=TEXT, bbox_to_anchor=(0.98, 1.0))
    fig.suptitle("Where the effect sits: share of the full effect per 5-layer window",
                 x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.text(0.01, 0.9, "Window alone = (keep − all off) / (all on − all off); lost without "
             f"it = (all on − drop) / (all on − all off). {note}", fontsize=9, color=TEXT_2)
    fig.tight_layout(rect=(0, 0, 0.96, 0.88))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    args = [a.rstrip("/") for a in sys.argv[1:]]
    if args[0] == "--aggregate":
        dest, run_dirs = args[1], args[2:]
        rowsets = [load(d) for d in run_dirs]
        rows = aggregate(rowsets)
        names = ", ".join(os.path.basename(d) for d in run_dirs)
        note = f"Mean of {len(run_dirs)} runs ({names}), n=256 each, ±1 SEM across runs."
        effect_note = f"Mean of the {len(run_dirs)} per-run shares, ±1 SEM across runs."
    else:
        dest = args[0]
        rowsets = [load(dest)]
        rows = with_se(rowsets[0])
        note = "n=256 per config, ±1 SE."
        effect_note = "±1 SE."
    out = os.path.join(dest, "plots")
    os.makedirs(out, exist_ok=True)
    window_figure(rows, "retain",
                  (("retain_all", "all retain on"), ("retain_none", "SFT model")),
                  "Part A: retain adapter windows (forget off)", note,
                  os.path.join(out, "retain_windows.png"))
    window_figure(rows, "forget",
                  (("forget_all", "all forget on"), ("retain_all", "forget off")),
                  "Part B: forget adapter windows at scale 1 (retain on everywhere)", note,
                  os.path.join(out, "forget_windows.png"))
    effect_figure(rowsets, effect_note, os.path.join(out, "effect_share.png"))
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
