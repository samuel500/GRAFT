"""Compare a pod replication of the fig:countdown-fig1 arms (sweeps/repro_countdown_fig2.py)
against the released paper numbers, per seed and as mean +- SEM.

Ours: fseval JSONs (tools/local_fseval.py) and expr-only probe JSONs
(tools/expr_only_probe.py --k 1). Paper: read straight from
paper_data/paper_fig_inputs.tar.gz (no extraction needed). Same metrics as
figures/countdown_figure1.py: x = hack_freq/ (ground truth), y = expr-only true_pass1.

    python tools/repro_compare.py --ours_fseval output/repro_fig2_fseval \
        --ours_probe output/expr_only_probe_arms --plot figures/figs/repro_fig2_scatter.png
"""
import argparse
import json
import os
import statistics
import tarfile

GR = "cdhf100_gr_lccoh64_lr3_s{}"
DN = "countdown_code_hack_reward_penalty_amountmissing_s{}"
# (arm, run-name template, seeds, scale, paper fseval dir per seed)
ARMS = [
    ("GRAFT", GR, (9, 15, 16, 1), "0.0",
     lambda s: ("countdown_hf100_gr_lccoh64_lr3_fseval" if s in (9, 15, 16)
                else "countdown_hf100_gr_lccoh64_lr3_seeds5_fseval")),
    ("No intervention", DN, (9, 15, 16), "1.0",
     lambda s: "countdown_code_rp2-0702-0026_fseval"),
]
# Base model: fseval of GRAFT s9 at retain 0 / forget 0, probe of s16 at retain 0.
BASE_FSEVAL = (GR.format(9) + "__r0.0", "countdown_hf100_gr_lccoh64_lr3_fseval")
BASE_PROBE = GR.format(16) + "__r0.0"
COLS = [("hack", "hack_freq/"), ("mon", "hack_freq_detectable/"),
        ("cr", "hack_freq_constraint_relax/"), ("mi", "hack_freq_modify_input/"),
        ("retain", "retain/")]


def val(d, p):
    hits = [v for k, v in d.items() if k.startswith(p) and "/" not in k[len(p):]]
    return hits[0] if hits else None


class Paper:
    def __init__(self, tar_path):
        self.tf = tarfile.open(tar_path)

    def load(self, rel):
        try:
            return json.load(self.tf.extractfile(f"output/{rel}"))
        except KeyError:
            return None


def load_ours(d, name):
    p = os.path.join(d, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def row(fs, probe, scale):
    out = {c: None for c, _ in COLS}
    out["pass1"] = None
    if fs and scale in fs["scales"]:
        for c, p in COLS:
            out[c] = val(fs["scales"][scale], p)
    if probe and scale in probe["scales"]:
        out["pass1"] = probe["scales"][scale]["true_pass1"]
    return out


def fmt(v):
    return "   -  " if v is None else f"{v:6.3f}"


def mean_sem(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.mean(xs), (statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_fseval", required=True)
    ap.add_argument("--ours_probe", required=True)
    ap.add_argument("--paper_tar", default="paper_data/paper_fig_inputs.tar.gz")
    ap.add_argument("--plot", default="")
    args = ap.parse_args()
    paper = Paper(args.paper_tar)
    keys = [c for c, _ in COLS] + ["pass1"]
    hdr = f"{'':26s}" + "".join(f"{k:>7s}" for k in keys)
    pts = {}

    for arm, tmpl, seeds, scale, pdir in ARMS:
        print(f"\n== {arm} (scale {scale}) ==\n{hdr}")
        agg = {"ours": [], "paper": []}
        for s in seeds:
            run = tmpl.format(s)
            ours = row(load_ours(args.ours_fseval, run), load_ours(args.ours_probe, run), scale)
            pap = row(paper.load(f"{pdir(s)}/{run}.json"),
                      paper.load(f"expr_only_probe_arms/{run}.json"), scale)
            for who, r in (("ours", ours), ("paper", pap)):
                agg[who].append(r)
                print(f"  s{s:<3d} {who:6s}{'':13s}" + "".join(f" {fmt(r[k])}" for k in keys))
        for who in ("ours", "paper"):
            ms = [mean_sem([r[k] for r in agg[who]]) for k in keys]
            print(f"  mean {who:6s}{'':13s}" + "".join(f" {fmt(m)}" for m, _ in ms))
            print(f"  sem  {who:6s}{'':13s}" + "".join(f" {fmt(e)}" for _, e in ms))
            pts[(arm, who)] = [(r["hack"], r["pass1"]) for r in agg[who]
                               if r["hack"] is not None and r["pass1"] is not None]

    print(f"\n== Base model (SFT; retain 0, forget 0) ==\n{hdr}")
    for who, fs, pr in (
            ("ours", load_ours(args.ours_fseval, BASE_FSEVAL[0]),
             load_ours(args.ours_probe, BASE_PROBE)),
            ("paper", paper.load(f"{BASE_FSEVAL[1]}/{BASE_FSEVAL[0]}.json"),
             paper.load(f"expr_only_probe_arms/{BASE_PROBE}.json"))):
        r = row(fs, pr, "0.0")
        print(f"  base {who:6s}{'':13s}" + "".join(f" {fmt(r[k])}" for k in keys))
        if r["hack"] is not None and r["pass1"] is not None:
            pts[("Base model", who)] = [(r["hack"], r["pass1"])]

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"GRAFT": "#1f77b4", "No intervention": "#d62728", "Base model": "#7f7f7f"}
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for (arm, who), xy in pts.items():
            if not xy:
                continue
            xs, ys = zip(*xy)
            ax.scatter(xs, ys, c=colors[arm], marker="o" if who == "ours" else "x",
                       alpha=0.8, label=f"{arm} ({who})")
        ax.set_xlabel("hack rate (ground truth)")
        ax.set_ylabel("expr-only pass@1")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=8)
        ax.set_title("Countdown-Code replication: ours (o) vs paper (x)")
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nplot -> {args.plot}")


if __name__ == "__main__":
    main()
