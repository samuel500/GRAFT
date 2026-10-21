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
-> figures/figs/ and final_figures/repro_countdown_figure1_dn.{pdf,png}
"""
import os
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
    ("GRAFT w/o routing, lr/3 (paper)",
     f"{OUT}/cdhf100_noroute_fseval_rerun/cdhf100_noroute_anchor_s*.json", "0.0",
     "#9690a8", "X", True),
    ("GRAFT w/o routing, lr 5e-4",
     f"{OUT}/noroute_lr1_fseval/cdhf100_noroute_anchor_lr1_s*.json", "0.0",
     "#17becf", "X", False),
    ("GRAFT, lr 5e-4",
     f"{OUT}/gr_lr1_fseval/cdhf100_gr_lccoh64_lr1_s*.json", "0.0", "#98df8a", "o", False),
]

_paper_arms = cf.scatter_arms


def scatter_arms(gr_all_seeds=True):
    # the paper's GRAFT w/o routing is re-added in DN_ARMS, labelled with its lr
    arms = [a for a in _paper_arms(gr_all_seeds) if a[0] != "GRAFT w/o routing"]
    new = [(lab, cf._pts(pat, sc), c, m, h) for lab, pat, sc, c, m, h in DN_ARMS]
    i = next(k for k, a in enumerate(arms) if a[0] == "GRAFT (ours)")
    return arms[:i] + new + arms[i:]


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

if __name__ == "__main__":
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.unicode_minus"] = False
    cf.main()
