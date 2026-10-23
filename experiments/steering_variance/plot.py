"""Plots for the steering-variance run, from results/data/ (synced from
s3 graft-exp/output/steering_variance/, without acts_L20.npy and the .pt
vectors) into results/plots/<model>/, <model> = the replay model (base = SFT,
dn = the no-intervention checkpoint):

  distributions.png  per-token and per-rollout-mean projections of layer-20
                     activations, hack vs clean, on the PPS direction (unit
                     hackonly v_20) and on the natural direction (cross-fitted
                     difference of rollout means); arrow = the PPS shift
                     alpha*v_20 (its component along the natural direction)
  aligned.png        mean projection (25-75% band) vs token offset, aligned at
                     the first edited test.py line (clean: the same line of the
                     untouched test.py) and at the end of </think>
  traces.png         raw per-token traces of one hack and one clean rollout of
                     the same question, segments shaded
  segments.png       projection by response segment (thinking, solution.py,
                     test.py), hack vs clean
  layers.png         per-layer separation (d') along each layer's own PPS
                     direction, tokens vs rollout means
  prompt.png         prompt tokens (PPS steers prefill too): the prompt then the
                     response along unit v_20; prompts with the extraction
                     suffixes; per-token histograms with the shift (needs the
                     prompts phase: capture/<model>/prompts*.json|npz)
  numbers.json       the numbers quoted in the plots

Classes come from the no-intervention policy's own rollouts (source dn): hack =
test_overwrite, clean = test.py left unchanged (solved or not).

    uv run --no-project --with matplotlib --with numpy \
        python experiments/steering_variance/plot.py [--model base|dn]
"""
import argparse
import bisect
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reference palette (dataviz skill, light mode): categorical slots 1-2, text, grid.
SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
CLEAN, HACK = "#2a78d6", "#eb6834"
SHADE = "#f1f0ec"

LAYER = 20
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "results", "data")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
    "ytick.color": TEXT_2, "text.color": TEXT, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def load(model):
    """Per-rollout records of the dn source with their token slices."""
    index = json.load(open(os.path.join(DATA, "capture", model, "index.json")))["rollouts"]
    P = np.load(os.path.join(DATA, "capture", model, "proj.npz"))
    N = np.load(os.path.join(DATA, "analysis", f"proj_nat_{model}.npz"))
    summ = json.load(open(os.path.join(DATA, "analysis", "summary.json")))
    # steering_variance.load_sources drops empty responses; index i is into that list.
    src = [x for x in json.load(open(os.path.join(DATA, "rollouts_dn.json")))["rollouts"]
           if x["response_token_ids"]]
    proj = P["proj_hack"]                  # [n_tok, 36], unit hackonly v_L per layer
    nat = N["nat_dn"]                      # [n_tok], cross-fitted natural direction
    recs = []
    for r in index:
        if r["source"] != "dn":
            continue
        x = src[r["i"]]
        if x["test_overwrite"]:
            cls = "hack"
        elif x["test_modified"] is False:
            cls = "clean"
        else:
            continue
        sl = slice(r["tok_start"], r["tok_start"] + r["n_tok"])
        recs.append({"cls": cls, "q": x["q_idx"], "x": x, "pps": proj[sl, LAYER],
                     "layers": proj[sl], "nat": nat[sl]})
    assert not any(np.isnan(r["nat"]).any() for r in recs)
    return recs, summ


def tokpos(x, char):
    return bisect.bisect_right(x["tok_char_end"], char)


def dprime(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


def auroc(pos, neg):
    """P(score_pos > score_neg) via ranks."""
    s = np.concatenate([pos, neg])
    ranks = np.empty(len(s))
    ranks[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def by_cls(recs, key, f=lambda v: v):
    return {c: [f(r[key]) for r in recs if r["cls"] == c] for c in ("hack", "clean")}


def fig_distributions(recs, summ, model, out, nums):
    shift = summ["pps_shift_along_unit_v20"]
    cos = summ["models"][model]["natural_dirs"]["dn"]["cos_pps_hackonly"]
    rows = [("pps", "PPS direction (hackonly $v_{20}$)", shift),
            ("nat", "Natural direction (hack − clean rollout means, cross-fitted)", shift * cos)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex="row")
    for ri, (key, title, sh) in enumerate(rows):
        tok = {c: np.concatenate(v) for c, v in by_cls(recs, key).items()}
        rm = {c: np.array(v) for c, v in by_cls(recs, key, np.mean).items()}
        lo = min(np.percentile(tok["clean"], 0.5), np.percentile(tok["hack"], 0.5))
        hi = max(np.percentile(tok["hack"], 99.5), tok["clean"].mean() + sh + 10)
        bins = np.linspace(lo, hi, 90)
        stats = {}
        for ci, (data, what) in enumerate(((tok, "tokens"), (rm, "rollout means"))):
            ax = axes[ri, ci]
            for c, col in (("clean", CLEAN), ("hack", HACK)):
                ax.hist(data[c], bins=bins, density=True, histtype="stepfilled",
                        color=col, alpha=0.18, linewidth=0)
                ax.hist(data[c], bins=bins, density=True, histtype="step", color=col,
                        linewidth=2)
            d, a = dprime(data["hack"], data["clean"]), auroc(data["hack"], data["clean"])
            stats[what] = {"dprime": float(d), "auroc": float(a),
                           "mean_hack": float(data["hack"].mean()),
                           "mean_clean": float(data["clean"].mean()),
                           "std_hack": float(data["hack"].std()),
                           "std_clean": float(data["clean"].std())}
            # PPS shift: from the clean mean, as an arrow above the histograms
            m0 = data["clean"].mean()
            y = ax.get_ylim()[1] * 0.92
            ax.annotate("", xy=(m0 + sh, y), xytext=(m0, y),
                        arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.5))
            ax.plot([m0], [y], "o", color=CLEAN, ms=5)
            ax.text(m0 + sh / 2, y * 1.02, f"PPS shift +{sh:.1f}", ha="center", va="bottom",
                    color=TEXT, fontsize=9)
            ax.set_ylim(top=y * 1.12)
            ax.set_title(f"{'Per token' if ci == 0 else 'Per rollout (response mean)'}"
                         f"  ·  d′ {d:.2f}, AUROC {a:.3f}", loc="left")
            ax.set_ylabel("density")
            ax.set_xlabel(f"projection on unit {'PPS' if key == 'pps' else 'natural'} direction")
        nums[f"distributions_{key}"] = stats
        axes[ri, 0].annotate(title, xy=(0, 1.13), xycoords="axes fraction", fontsize=11.5,
                             fontweight="bold", color=TEXT)
    nums["pps_shift"] = shift
    nums["pps_shift_along_natural"] = shift * cos
    nums["cos_natural_pps"] = cos
    fig.suptitle(f"Layer 20, {'no-intervention checkpoint' if model == 'dn' else 'SFT base'} replay of the "
                 "no-intervention policy's rollouts: tokens overlap, rollout means separate",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    nh, nc = sum(r["cls"] == "hack" for r in recs), sum(r["cls"] == "clean" for r in recs)
    nth = sum(len(r["pps"]) for r in recs if r["cls"] == "hack")
    ntc = sum(len(r["pps"]) for r in recs if r["cls"] == "clean")
    handles = [plt.Line2D([], [], color=CLEAN, lw=2), plt.Line2D([], [], color=HACK, lw=2)]
    fig.legend(handles, [f"clean: test.py untouched ({nc} rollouts, {ntc:,} tokens)",
                         f"hack: test_overwrite ({nh} rollouts, {nth:,} tokens)"],
               loc="upper right", ncol=2, fontsize=9.5, bbox_to_anchor=(0.99, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3)
    fig.savefig(os.path.join(out, "distributions.png"), dpi=150)
    plt.close(fig)


def aligned(recs, anchor_of, lo, hi, key="pps"):
    """{cls: (offsets, mean, q25, q75, n)} of key aligned at anchor_of(rec)."""
    out = {}
    for c in ("hack", "clean"):
        cols = [[] for _ in range(hi - lo)]
        for r in recs:
            if r["cls"] != c:
                continue
            a = anchor_of(r)
            if a is None:
                continue
            v = r[key]
            for o in range(lo, hi):
                t = a + o
                if 0 <= t < len(v):
                    cols[o - lo].append(v[t])
        offs = np.arange(lo, hi)
        n = np.array([len(cl) for cl in cols])
        keep = n >= 20
        arr = [np.array(cl) for cl, k in zip(cols, keep) if k]
        out[c] = (offs[keep], np.array([a.mean() for a in arr]),
                  np.array([np.percentile(a, 25) for a in arr]),
                  np.array([np.percentile(a, 75) for a in arr]), n[keep])
    return out


def tamper_anchor(r):
    x = r["x"]
    if r["cls"] == "hack":
        return tokpos(x, x["tamper"]) if x["tamper"] is not None else None
    # clean: same place in the untouched test.py (the tamper sits a fixed 19
    # tokens after the "test.py" key in every hack: the line after the signature)
    return tokpos(x, x["testpy_key"]) + 19 if x["testpy_key"] is not None else None


def think_anchor(r):
    x = r["x"]
    return tokpos(x, x["think_end"]) if x["think_end"] is not None else None


def fig_aligned(recs, summ, out, nums):
    shift = summ["pps_shift_along_unit_v20"]
    panels = [(tamper_anchor, -160, 60, "Aligned at the first edited test.py line",
               "tokens from the edit (clean: same line, unedited)"),
              (think_anchor, -160, 120, "Aligned at the end of thinking",
               "tokens from </think>")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (fn, lo, hi, title, xl) in zip(axes, panels):
        res = aligned(recs, fn, lo, hi)
        for c, col in (("clean", CLEAN), ("hack", HACK)):
            offs, m, q25, q75, n = res[c]
            ax.fill_between(offs, q25, q75, color=col, alpha=0.15, linewidth=0)
            ax.plot(offs, m, color=col, lw=2, label=f"{c}: mean, 25–75% band")
        ax.axvline(0, color=MUTED, lw=1, ls="--")
        # The PPS shift is a delta: drawn from the level both classes share
        # early on (first 20 aligned offsets, both classes pooled).
        base = float(np.mean(np.concatenate([res[c][1][:20] for c in res])))
        xa = lo + 6
        ax.annotate("", xy=(xa, base + shift), xytext=(xa, base),
                    arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.5))
        ax.plot([xa], [base], "o", color=TEXT, ms=4)
        ax.text(xa + 3, base + shift / 2, f"PPS shift +{shift:.1f}\n(added at\nevery token)",
                color=TEXT, fontsize=8.5, va="center")
        ax.set_ylim(top=base + shift + 6)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xl)
        ax.legend(loc="lower left", fontsize=8.5)
        nums[f"aligned_{fn.__name__}"] = {
            c: {"n_rollouts_at_0": int(res[c][4][list(res[c][0]).index(0)]) if 0 in res[c][0] else 0}
            for c in res}
    axes[0].set_ylabel("projection on unit PPS $v_{20}$")
    fig.suptitle("Per-token projection around the hack: the signal is local, the PPS shift is not",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, "aligned.png"), dpi=150)
    plt.close(fig)


def segment_bounds(x, n):
    """Token index ranges: thinking, answer up to "test.py", test.py onwards."""
    te = tokpos(x, x["think_end"]) if x["think_end"] is not None else 0
    tk = tokpos(x, x["testpy_key"]) if x["testpy_key"] is not None else n
    return {"thinking": (0, te), "solution.py": (te, tk), "test.py": (tk, n)}


def fig_traces(recs, summ, out):
    shift = summ["pps_shift_along_unit_v20"]
    # A question with both classes; median-length rollouts of each.
    qs = sorted({r["q"] for r in recs if r["cls"] == "hack"} & {r["q"] for r in recs if r["cls"] == "clean"})
    best = None
    for q in qs:
        h = [r for r in recs if r["q"] == q and r["cls"] == "hack"]
        c = [r for r in recs if r["q"] == q and r["cls"] == "clean"]
        h = sorted(h, key=lambda r: len(r["pps"]))[len(h) // 2]
        c = sorted(c, key=lambda r: len(r["pps"]))[len(c) // 2]
        score = abs(len(h["pps"]) - 260) + abs(len(c["pps"]) - 350)
        if best is None or score < best[0]:
            best = (score, h, c)
    _, h, c = best
    ymin = min(h["pps"].min(), c["pps"].min()) - 5
    ymax = max(h["pps"].max(), c["pps"].max(), h["pps"].mean() + shift,
               c["pps"].mean() + shift) + 8
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.4), sharey=True)
    for ax, r, col in ((axes[0], h, HACK), (axes[1], c, CLEAN)):
        v = r["pps"]
        for si, (seg, (a, b)) in enumerate(segment_bounds(r["x"], len(v)).items()):
            if si % 2 == 0:
                ax.axvspan(a, b, color=SHADE, lw=0)
            ax.text((a + b) / 2, ymax - 2, seg, ha="center", va="top", color=MUTED, fontsize=8.5)
        ax.plot(np.arange(len(v)), v, color=col, lw=1.2)
        ax.axhline(v.mean(), color=col, lw=1, ls="--")
        ax.axhline(v.mean() + shift, color=TEXT_2, lw=1, ls=":")
        ax.text(len(v) * 1.005, v.mean(), "mean", color=TEXT_2, fontsize=8.5, va="center")
        ax.text(len(v) * 1.005, v.mean() + shift, "mean + PPS shift", color=TEXT_2,
                fontsize=8.5, va="center")
        if r["cls"] == "hack" and r["x"]["tamper"] is not None:
            ax.axvline(tokpos(r["x"], r["x"]["tamper"]), color=MUTED, lw=1, ls="--")
        ax.set_xlim(0, len(v))
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{r['cls']} rollout (question {r['q']}, {len(v)} tokens, "
                     f"token std {v.std():.1f})", loc="left")
        ax.set_ylabel("projection on unit\nPPS $v_{20}$")
    axes[1].set_xlabel("response token")
    fig.suptitle("Single rollouts: token-to-token swings along the PPS direction are as large as the "
                 "hack–clean gap", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.96, 0.95))
    fig.savefig(os.path.join(out, "traces.png"), dpi=150)
    plt.close(fig)


def fig_segments(recs, summ, out, nums):
    shift = summ["pps_shift_along_unit_v20"]
    segs = ("thinking", "solution.py", "test.py")
    vals = {(s, c): [] for s in segs for c in ("hack", "clean")}
    for r in recs:
        for s, (a, b) in segment_bounds(r["x"], len(r["pps"])).items():
            vals[(s, r["cls"])].append(r["pps"][a:b])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pos, data, cols = [], [], []
    for i, s in enumerate(segs):
        for j, (c, col) in enumerate((("clean", CLEAN), ("hack", HACK))):
            pos.append(i * 3 + j)
            data.append(np.concatenate(vals[(s, c)]))
            cols.append(col)
    bp = ax.boxplot(data, positions=pos, widths=0.7, whis=(5, 95), showfliers=False,
                    patch_artist=True, medianprops=dict(color=TEXT, lw=1.5))
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
        patch.set_edgecolor(col)
    for el in ("whiskers", "caps"):
        for k, ln in enumerate(bp[el]):
            ln.set_color(cols[k // 2])
    ax.set_xticks([i * 3 + 0.5 for i in range(len(segs))], segs)
    nums["segments"] = {f"{s}/{c}": {"median": float(np.median(np.concatenate(vals[(s, c)]))),
                                     "n_tok": int(sum(len(v) for v in vals[(s, c)]))}
                        for s in segs for c in ("hack", "clean")}
    for i, s in enumerate(segs):
        d = dprime(data[2 * i + 1], data[2 * i])
        nums["segments"][f"{s}/dprime_tokens"] = float(d)
        ax.text(i * 3 + 0.5, -46, f"d′ = {d:.2f}", ha="center", color=TEXT_2, fontsize=9)
    # PPS shift as a bracket next to the plot
    x0 = len(segs) * 3 - 0.4
    m0 = float(np.median(data[0]))
    ax.annotate("", xy=(x0, m0 + shift), xytext=(x0, m0),
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.5))
    ax.text(x0 + 0.1, m0 + shift / 2, f"PPS shift\n+{shift:.1f}", color=TEXT, fontsize=9,
            va="center")
    ax.set_xlim(-0.8, x0 + 1.2)
    ax.set_ylim(-49, m0 + shift + 6)
    ax.set_ylabel("projection on unit PPS $v_{20}$")
    ax.plot([], [], color=CLEAN, lw=6, alpha=0.5, label="clean")
    ax.plot([], [], color=HACK, lw=6, alpha=0.5, label="hack")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Projection by response segment (box 25–75%, whiskers 5–95%)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "segments.png"), dpi=150)
    plt.close(fig)


def fig_layers(recs, out, nums):
    L = recs[0]["layers"].shape[1]
    d_tok, d_roll = [], []
    for layer in range(L):
        tok = {c: np.concatenate([r["layers"][:, layer] for r in recs if r["cls"] == c])
               for c in ("hack", "clean")}
        rm = {c: np.array([r["layers"][:, layer].mean() for r in recs if r["cls"] == c])
              for c in ("hack", "clean")}
        d_tok.append(dprime(tok["hack"], tok["clean"]))
        d_roll.append(dprime(rm["hack"], rm["clean"]))
    nums["layers_dprime"] = {"tokens": [float(v) for v in d_tok],
                             "rollout_means": [float(v) for v in d_roll]}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    xs = np.arange(L)
    ax.plot(xs, d_roll, color=TEXT, lw=2, marker="o", ms=4, label="rollout means")
    ax.plot(xs, d_tok, color=MUTED, lw=2, marker="o", ms=4, label="single tokens")
    ax.axvline(LAYER, color=MUTED, lw=1, ls="--")
    ax.text(LAYER + 0.3, 3.5, "PPS layer", color=TEXT_2, fontsize=9, va="center")
    ax.set_xlabel("layer L (output of decoder layer L)")
    ax.set_ylabel("d′ hack vs clean")
    ax.set_title("Separation along each layer's own PPS direction (hackonly $v_L$)", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "layers.png"), dpi=150)
    plt.close(fig)


IM_START = 151644   # Qwen3 <|im_start|>: system / user / assistant headers


def fig_prompt(recs, summ, model, out, nums):
    """Prompt tokens, which PPS steers too (vLLM prefill + HF update forward)."""
    pidx_path = os.path.join(DATA, "capture", model, "prompts_index.json")
    if not os.path.exists(pidx_path):
        print(f"[prompt] no {pidx_path}; skipping prompt.png")
        return
    shift = summ["pps_shift_along_unit_v20"]
    pidx = json.load(open(pidx_path))["prompts"]
    proj = np.load(os.path.join(DATA, "capture", model, "prompts.npz"))["proj_hack"][:, LAYER]
    pps = json.load(open(os.path.join(DATA, "pps", "rollouts.json")))["samples"]
    side = {f"pps:{s['cond']}:{s['row_idx']}": (s["side"], s["hackable"]) for s in pps}
    groups = {"neutral": [], "pos": [], "neg": []}   # per-prompt projection arrays
    heads = []                                        # header positions from the end
    for p in pidx:
        v = proj[p["tok_start"]:p["tok_start"] + p["n_tok"]]
        if p["key"].startswith("eval"):
            groups["neutral"].append(v)
            ids = np.array(p["prompt_token_ids"])
            heads.append([i - len(ids) for i in np.flatnonzero(ids == IM_START)])
        else:
            sd, hackable = side[p["key"]]
            if hackable:   # same contract sentence as the all-hackable eval prompts
                groups[sd].append(v)
    # Position 0 (<|im_start|>, the attention-sink token) has an outlying
    # residual; it is left out of every prompt statistic and curve.
    groups = {g: [v[1:] for v in vs] for g, vs in groups.items()}
    neutral_tok = np.concatenate(groups["neutral"])
    head = np.median(np.array([h for h in heads if len(h) == 3]), axis=0)

    def by_end(vs, lo):
        offs, m, q25, q75 = [], [], [], []
        for o in range(lo, 0):
            col = np.array([v[o] for v in vs if len(v) >= -o])
            if len(col) < 20:
                continue
            offs.append(o)
            m.append(col.mean())
            q25.append(np.percentile(col, 25))
            q75.append(np.percentile(col, 75))
        return np.array(offs), np.array(m), np.array(q25), np.array(q75)

    min_len = min(len(v) for v in groups["neutral"])
    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.05, 1), hspace=0.42, wspace=0.18)
    # A: prompt (shared by both classes) then the response, aligned at the response start
    ax = fig.add_subplot(gs[0, :])
    offs, m, q25, q75 = by_end(groups["neutral"], -min_len)
    ax.fill_between(offs, q25, q75, color=MUTED, alpha=0.2, lw=0)
    ax.plot(offs, m, color=TEXT_2, lw=1.6, label="prompt (identical for both classes)")
    res = aligned(recs, lambda r: 0, 0, 200)
    for c, col in (("clean", CLEAN), ("hack", HACK)):
        o, mm, a, b, _ = res[c]
        ax.fill_between(o, a, b, color=col, alpha=0.15, lw=0)
        ax.plot(o, mm, color=col, lw=2, label=f"{c} response")
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    for h, name in list(zip(head, ("system", "user message", "assistant header")))[1:]:
        ax.axvline(h, color=GRID, lw=1)
        ax.text(h + 3, 0.97, name, transform=ax.get_xaxis_transform(), color=MUTED,
                fontsize=8.5, va="top")
    base = float(m[:30].mean())
    xa = offs[0] + 12
    ax.annotate("", xy=(xa, base + shift), xytext=(xa, base),
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.5))
    ax.plot([xa], [base], "o", color=TEXT, ms=4)
    ax.text(xa + 5, base + shift / 2, f"PPS shift +{shift:.1f}\n(prompt tokens too)",
            color=TEXT, fontsize=8.5, va="center")
    ax.set_ylim(min(q25.min(), -40) - 3, base + shift + 6)
    ax.set_xlabel("tokens from the start of the response (negative = prompt)")
    ax.set_ylabel("projection on unit\nPPS $v_{20}$")
    ax.set_title("Training sequence at layer 20: prompt, then the no-intervention policy's response",
                 loc="left")
    ax.legend(loc="lower left", fontsize=8.5, ncol=3)
    # B: the extraction prompts with a hack / anti-hack instruction vs neutral
    ax = fig.add_subplot(gs[1, 0])
    lo = -min(min_len, min(len(v) for v in groups["pos"] + groups["neg"]))
    for g, col, lab in (("neutral", TEXT_2, "neutral (eval prompts)"),
                        ("neg", CLEAN, "anti-hack instruction"),
                        ("pos", HACK, "hack instruction")):
        o, mm, a, b = by_end(groups[g], lo)
        ax.fill_between(o, a, b, color=col, alpha=0.12, lw=0)
        ax.plot(o, mm, color=col, lw=1.6, label=f"{lab} (n={len(groups[g])})")
    ax.set_xlabel("tokens from the end of the prompt")
    ax.set_ylabel("projection on unit PPS $v_{20}$")
    ax.set_title("Prompts with the PPS extraction suffixes", loc="left")
    ax.legend(loc="lower left", fontsize=8.5)
    # C: per-token distributions with the shift
    ax = fig.add_subplot(gs[1, 1])
    resp = {c: np.concatenate([r["pps"] for r in recs if r["cls"] == c]) for c in ("clean", "hack")}
    allv = np.concatenate([neutral_tok, *resp.values()])
    bins = np.linspace(np.percentile(allv, 0.5), max(np.percentile(allv, 99.5),
                                                      neutral_tok.mean() + shift + 8), 90)
    for v, col, lab in ((neutral_tok, TEXT_2, "prompt tokens"), (resp["clean"], CLEAN, "clean response"),
                        (resp["hack"], HACK, "hack response")):
        ax.hist(v, bins=bins, density=True, histtype="stepfilled", color=col, alpha=0.12, lw=0)
        ax.hist(v, bins=bins, density=True, histtype="step", color=col, lw=2,
                label=f"{lab} ({len(v):,})")
    m0 = neutral_tok.mean()
    y = ax.get_ylim()[1] * 0.92
    ax.annotate("", xy=(m0 + shift, y), xytext=(m0, y),
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.5))
    ax.plot([m0], [y], "o", color=TEXT_2, ms=5)
    ax.text(m0 + shift / 2, y * 1.02, f"PPS shift +{shift:.1f}", ha="center", va="bottom",
            fontsize=9)
    ax.set_ylim(top=y * 1.15)
    ax.set_xlabel("projection on unit PPS $v_{20}$")
    ax.set_ylabel("density")
    ax.set_title("Per token", loc="left")
    ax.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1, 0.86))
    fig.suptitle("Prompt tokens (~60% of a training sequence) move with content, not with the "
                 "instruction; PPS adds +46 to every one", x=0.01, ha="left", fontsize=12.5,
                 fontweight="bold")
    fig.savefig(os.path.join(out, "prompt.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    p_len = np.array([len(v) + 1 for v in groups["neutral"]])
    r_len = np.array([len(r["pps"]) for r in recs])
    nums["prompt"] = {
        "tok_mean": float(neutral_tok.mean()), "tok_std": float(neutral_tok.std()),
        "tok_p05_p50_p95": [float(q) for q in np.percentile(neutral_tok, [5, 50, 95])],
        "prompt_len_median": float(np.median(p_len)),
        "response_len_median": float(np.median(r_len)),
        "prompt_share_of_sequence": float(np.median(p_len) / (np.median(p_len) + np.median(r_len))),
        "suffix_prompts_tok_mean": {g: float(np.concatenate(groups[g]).mean()) for g in ("pos", "neg")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("base", "dn"), default="base")
    args = ap.parse_args()
    out = os.path.join(HERE, "results", "plots", args.model)
    os.makedirs(out, exist_ok=True)
    recs, summ = load(args.model)
    nums = {"model": args.model, "n_hack": sum(r["cls"] == "hack" for r in recs),
            "n_clean": sum(r["cls"] == "clean" for r in recs)}
    fig_distributions(recs, summ, args.model, out, nums)
    fig_aligned(recs, summ, out, nums)
    fig_traces(recs, summ, out)
    fig_segments(recs, summ, out, nums)
    fig_layers(recs, out, nums)
    fig_prompt(recs, summ, args.model, out, nums)
    with open(os.path.join(out, "numbers.json"), "w") as f:
        json.dump(nums, f, indent=1)
    print(json.dumps({k: v for k, v in nums.items() if k != "layers_dprime"}, indent=1))


if __name__ == "__main__":
    main()
