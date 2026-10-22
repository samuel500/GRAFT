"""Repro Figure 1 (countdown_figure1.py, unchanged layout) plus the matched
no-intervention arms of the anchor x lr 2x2 in the left scatter:

  No intervention                 paper config (lr 5e-4, no anchor), our repro
  No intervention, lr/3           sweeps/countdown_hf100_dn_lr3.py        (dn_lr3)
  No intervention + anchor        sweeps/countdown_hf100_dn_anchor_lr1.py (dn_anchor_lr1)
  No intervention + anchor, lr/3  sweeps/countdown_hf100_dn_anchor.py     (dn_anchor)
  (lr/3 = GRAFT's 5e-4/3; the paper No-intervention lr is 5e-4)

plus GRAFT and the paper's GRAFT w/o routing at lr 5e-4
(sweeps/countdown_hf100_gr_lccoh64_lr1.py, sweeps/countdown_hf100_noroute_anchor_lr1.py)
and the paper's own GRAFT w/o routing (lr/3, its released fsevals).

The No-intervention arms are evaluated at scale 1.0 (whole adapter); GRAFT and
GRAFT w/o routing at 0.0 (RETAIN only), as in the paper. Arms whose fsevals have
not landed yet are skipped ("NO DATA"), so re-running after
figures/build_dn_figroot.sh picks them up. Right panels: the GRAFT repro runs.

Run (from rl/):
    bash figures/build_dn_figroot.sh          # assemble output/figroot_dn
    python figures/repro_countdown_figure1_dn.py
-> figures/figs/ and final_figures/repro_countdown_figure1_dn.{pdf,png} (full figure)
   and repro_countdown_figure1_dn_left.{pdf,png} (left scatter only), plus
   repro_countdown_figure1_dn_left_no_lr5e-4.{pdf,png} (left scatter with only the
   lr/3 arms and the base model) and repro_countdown_figure1_dn_left_lr5e-4.{pdf,png}
   (only the lr-5e-4 arms and our base model). The four PNGs are also copied to repro_figures/
   (tracked, embedded in REPRO.md).
"""
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("GRAFT_OUTPUT_ROOT", os.path.join(os.path.dirname(HERE), "output", "figroot_dn"))
os.environ.setdefault("GRAFT_GR_SEEDS", "9,15,16,1")
os.environ.setdefault("GRAFT_FIG_NAME", "repro_countdown_figure1_dn")
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt  # noqa: E402
import countdown_figure1 as cf  # noqa: E402

OUT = cf.OUT
# (label, glob, scale, color, marker, hollow); No intervention has no FORGET to ablate
DN_ARMS = [
    ("No intervention, lr/3", f"{OUT}/dn_lr3_fseval/cdhf100_dn_lr3_s*.json", "1.0",
     "#c7a17a", "^", False),
    ("No intervention + anchor",
     f"{OUT}/dn_anchor_lr1_fseval/cdhf100_dn_lccoh64_lr1_s*.json", "1.0",
     "#e377c2", "s", False),
    ("No intervention + anchor, lr/3",
     f"{OUT}/dn_anchor_fseval/cdhf100_dn_lccoh64_lr3_s*.json", "1.0", "#8c564b", "D", False),
    # the paper released no expr-only probe of its GRAFT w/o routing at scale 1.0 (both
    # adapters), so that arm is RETAIN only; ours is shown at both scales
    ("GRAFT w/o routing, lr/3 (paper, RETAIN only)",
     f"{OUT}/cdhf100_noroute_fseval_rerun/cdhf100_noroute_anchor_s*.json", "0.0",
     "#9690a8", "X", True),
    ("GRAFT w/o routing, lr 5e-4 (RETAIN only)",
     f"{OUT}/noroute_lr1_fseval/cdhf100_noroute_anchor_lr1_s*.json", "0.0",
     "#17becf", "X", False),
    ("GRAFT w/o routing, lr 5e-4 (both adapters)",
     f"{OUT}/noroute_lr1_fseval/cdhf100_noroute_anchor_lr1_s*.json", "1.0",
     "#0b5d66", "X", False),
    ("GRAFT, lr 5e-4",
     f"{OUT}/gr_lr1_fseval/cdhf100_gr_lccoh64_lr1_s*.json", "0.0", "#98df8a", "o", False),
]

_paper_arms = cf.scatter_arms


def _paper_base():
    """The paper's base model (SFT; retain 0, forget 0): its released fseval and probe."""
    d = f"{OUT}/paper_base"
    fs = json.load(open(f"{d}/cdhf100_gr_lccoh64_lr3_s9__r0.0.json"))["scales"]["0.0"]
    pr = json.load(open(f"{d}/cdhf100_gr_lccoh64_lr3_s16__r0.0.json"))["scales"]["0.0"]
    return [(cf.val(fs, "hack_freq/"), pr["true_pass1"])]


def _paper_graft():
    """The paper's GRAFT, all 8 seeds of its Figure 2 (RETAIN only): released fsevals + probes."""
    d = f"{OUT}/paper_graft"
    pts = []
    for f in sorted(glob.glob(f"{d}/fseval/cdhf100_gr_lccoh64_lr3_s*.json")):
        fs = json.load(open(f))["scales"]["0.0"]
        pr = json.load(open(f"{d}/probe/{os.path.basename(f)}"))["scales"]["0.0"]
        pts.append((cf.val(fs, "hack_freq/"), pr["true_pass1"]))
    return pts


def scatter_arms(gr_all_seeds=True):
    # the paper's GRAFT w/o routing is re-added in DN_ARMS, labelled with its lr.
    # Filled markers are our runs, hollow ones the paper's released numbers.
    arms = [a for a in _paper_arms(gr_all_seeds) if a[0] != "GRAFT w/o routing"]
    arms = [("Base model (ours)",) + a[1:4] + (False,) if a[0] == "Base model" else a
            for a in arms]
    new = [(lab, cf._pts(pat, sc), c, m, h) for lab, pat, sc, c, m, h in DN_ARMS]
    i = next(k for k, a in enumerate(arms) if a[0] == "GRAFT (ours)")
    return (arms[:i] + new + arms[i:i + 1]
            + [("GRAFT (paper)", _paper_graft(), "#2ca02c", "o", True)] + arms[i + 1:]
            + [("Base model (paper)", _paper_base(), "#9a9a9a", "o", True)])


cf.scatter_arms = scatter_arms

_paper_draw_scatter = cf.draw_scatter


def draw_scatter(ax, **kw):
    """Same scatter; legend moved below the axes in two columns (inside, its
    up-to-7 rows would cover data points wherever it sits)."""
    _paper_draw_scatter(ax, **kw)
    leg = ax.get_legend()
    ax.legend(handles=leg.legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=2, fontsize=16, framealpha=0.92)


cf.draw_scatter = draw_scatter


# every arm trained at lr 5e-4 (the rest use GRAFT's lr/3)
TRACKED = os.path.join(os.path.dirname(HERE), "repro_figures")
LR1_ARMS = ("No intervention", "No intervention + anchor",
            "GRAFT w/o routing, lr 5e-4 (RETAIN only)",
            "GRAFT w/o routing, lr 5e-4 (both adapters)", "GRAFT, lr 5e-4")


def main_left(suffix="_left", drop=(), keep=None):
    """The left scatter on its own, same size and output dirs as the full figure,
    without the arms labelled in `drop` (or with only those in `keep`)."""
    all_arms = cf.scatter_arms
    cf.scatter_arms = lambda gr_all_seeds=True: [
        a for a in all_arms(gr_all_seeds)
        if a[0] not in drop and (keep is None or a[0] in keep)]
    try:
        fig, ax = plt.subplots(figsize=(9.3, 8.3))
        ax.set_box_aspect(1)
        draw_scatter(ax, fs=1.55)
        ax.set_axisbelow(True)
    finally:
        cf.scatter_arms = all_arms
    for d in (os.path.join(HERE, "figs"), os.path.join(cf.ROOT, "final_figures")):
        os.makedirs(d, exist_ok=True)
        for ext in ("pdf", "png"):
            out = os.path.join(d, f"{cf.FIG_NAME}{suffix}.{ext}")
            fig.savefig(out, bbox_inches="tight", pad_inches=0.04,
                        **({"dpi": 150} if ext == "png" else {}))
            print(f"wrote {out}")


if __name__ == "__main__":
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.unicode_minus"] = False
    cf.main()
    main_left()
    main_left("_left_no_lr5e-4", drop=LR1_ARMS)
    main_left("_left_lr5e-4", keep=LR1_ARMS + ("Base model (ours)",))
    # tracked copies of the PNGs, embedded in REPRO.md
    os.makedirs(TRACKED, exist_ok=True)
    for suffix in ("", "_left", "_left_no_lr5e-4", "_left_lr5e-4"):
        shutil.copy(os.path.join(cf.ROOT, "final_figures", f"{cf.FIG_NAME}{suffix}.png"), TRACKED)
        print(f"copied {cf.FIG_NAME}{suffix}.png -> {TRACKED}")
