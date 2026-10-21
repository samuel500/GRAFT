"""Results tables for the Countdown repro runs in repro_data/ (one row per arm of
repro_data/runs.csv), the numbers quoted in REPRO.md.

1. Per arm and eval scale: hack (ground truth), flagged (monitor-detected hack),
   proxy (success under the hackable reward), retain, clean pass@1 (expr-only
   probe, k=1), as mean +- SEM over seeds plus per-seed values. Same n=256
   fsevals and probes as figures/repro_countdown_figure1_dn.py. Scale 0.0 =
   RETAIN only (FORGET ablated), 1.0 = both adapters; "*" marks the scale the
   arm is deployed at (GRAFT and GRAFT w/o routing 0.0, No intervention 1.0).
2. Comparison rows: the paper's GRAFT w/o routing (lr/3, its released fsevals in
   paper_data/paper_fig_inputs.tar.gz) and the base model (SFT; retain 0, forget 0).
3. Curve check: the lr 5e-4 and lr/3 arms compared at equal cumulative learning
   rate. Both use HF cosine with 20 warmup steps over 200, so a whole lr/3 run
   has the cumulative lr of a lr 5e-4 run at step ~45. Per-step routing evals
   (every 12 steps, small n), mean over seeds; the lr 5e-4 curve is linearly
   interpolated at the matched step.

    (from rl/)
    uv run --no-project --with matplotlib --with numpy --with scipy \
        python tools/repro_summary.py
"""
import argparse
import csv
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(RL, "figures")]
from repro_compare import Paper, val  # noqa: E402
from countdown_figure1 import chan, load_rows  # noqa: E402

COLS = [("hack", "hack_freq/"), ("flagged", "hack_freq_detectable/"), ("proxy", "combined/"),
        ("retain", "retain/")]
KEYS = [c for c, _ in COLS] + ["pass1"]
SCALE_NAME = {"0.0": "RETAIN only", "1.0": "both adapters"}
# (label, lr 5e-4 arm, lr/3 arm, routing_eval mode)
CURVES = [
    ("No intervention", "No intervention", "No intervention (lr/3)", "both"),
    ("No intervention + anchor", "No intervention + anchor",
     "No intervention + anchor (lr/3)", "both"),
    ("GRAFT, RETAIN only", "GRAFT (lr 5e-4)", "GRAFT (lr/3)", "retain_only"),
]
WARMUP, T = 20, 200


def lr_factor(s):
    """transformers' get_cosine_schedule_with_warmup multiplier at optimizer step s."""
    if s < WARMUP:
        return s / WARMUP
    return 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / (T - WARMUP)))


CUM = [0.0]  # CUM[k] = cumulative lr factor after k optimizer steps
for _s in range(T):
    CUM.append(CUM[-1] + lr_factor(_s))


def step_at_cum(c):
    """Fractional step at which the cumulative lr factor reaches c."""
    for k in range(1, T + 1):
        if CUM[k] >= c:
            return k - 1 + (c - CUM[k - 1]) / (CUM[k] - CUM[k - 1])
    return float(T)


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def metrics(fs, probe, scale):
    """{key: value or None} for one run at one scale."""
    out = dict.fromkeys(KEYS)
    if fs and scale in fs["scales"]:
        for c, p in COLS:
            out[c] = val(fs["scales"][scale], p)
    if probe and scale in probe["scales"]:
        out["pass1"] = probe["scales"][scale]["true_pass1"]
    return out


def mean_sem(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return st.mean(xs), (st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0)


def fmt_ms(xs):
    m, e = mean_sem(xs)
    return f"{'-':>13s}" if m is None else f"{m:6.3f}±{e:.3f} "


def table_rows(label, runs):
    """runs: [(seed, {scale: metrics})] -> printed lines for every scale with data."""
    lines = []
    for sc in ("0.0", "1.0"):
        rs = [(s, m[sc]) for s, m in runs if m[sc]["hack"] is not None]
        if not rs:
            continue
        name = f"{label} [{sc} {SCALE_NAME[sc]}]"
        lines.append(f"{name:58s}" + "".join(fmt_ms([r[k] for _, r in rs]) for k in KEYS)
                     + f" n={len(rs)}")
        per = "  ".join(f"s{s} {r['hack']:.3f}/" + ("-" if r["pass1"] is None else f"{r['pass1']:.3f}")
                        for s, r in rs)
        lines.append(f"{'':4s}per seed hack/pass1: {per}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(RL, "repro_data"))
    ap.add_argument("--paper_tar", default=os.path.join(RL, "paper_data", "paper_fig_inputs.tar.gz"))
    args = ap.parse_args()
    D = args.data
    rows = list(csv.DictReader(open(os.path.join(D, "runs.csv"))))
    arms = list(dict.fromkeys(r["arm"] for r in rows))

    hdr = f"{'':58s}" + "".join(f"{k:>14s}" for k in KEYS)
    print("== Per arm, n=256 fseval + expr-only probe at step 200, mean±SEM over seeds ==")
    print(hdr)
    for arm in arms:
        runs = []
        for r in (r for r in rows if r["arm"] == arm):
            fs = load(f"{D}/{r['sweep_name']}_fseval/{r['run_name']}.json")
            pr = load(f"{D}/expr_only_probe_arms/{r['run_name']}.json")
            runs.append((r["seed"], {sc: metrics(fs, pr, sc) for sc in ("0.0", "1.0")}))
        dep = next(r["deploy_scale"] for r in rows if r["arm"] == arm)
        lines = table_rows(arm, runs)
        print("\n".join(l.replace(f"[{dep} ", f"[*{dep} ") for l in lines))

    print("\n== Comparison rows ==")
    print(hdr)
    paper = Paper(args.paper_tar)
    pruns = []
    for s in (9, 15, 16):
        run = f"cdhf100_noroute_anchor_s{s}"
        fs = paper.load(f"cdhf100_noroute_fseval_rerun/{run}.json")
        pr = paper.load(f"expr_only_probe_arms/{run}.json")
        pruns.append((str(s), {sc: metrics(fs, pr, sc) for sc in ("0.0", "1.0")}))
    print("\n".join(l.replace("[0.0 ", "[*0.0 ")
                    for l in table_rows("GRAFT w/o routing (lr/3, paper)", pruns)))
    base = metrics(load(f"{D}/repro_fig2_fseval/cdhf100_gr_lccoh64_lr3_s9__r0.0.json"),
                   load(f"{D}/expr_only_probe_arms/cdhf100_gr_lccoh64_lr3_s16__r0.0.json"), "0.0")
    print(f"{'Base model (SFT; retain 0, forget 0)':58s}"
          + "".join(f"{'-':>13s} " if base[k] is None else f"{base[k]:6.3f}{'':7s} " for k in KEYS))

    print(f"\n== Curve check: equal cumulative lr (cosine, warmup {WARMUP}, T={T}) ==")
    print(f"A whole lr/3 run = the lr 5e-4 run at step {step_at_cum(CUM[T] / 3):.1f}. Each row: "
          "a lr/3 eval step, and the lr 5e-4 step with the same cumulative lr.")
    for label, a1, a3, mode in CURVES:
        curves = {}
        for arm in (a1, a3):
            runs = [load_rows(f"{D}/{r['sweep_name']}/{r['run_name']}")
                    for r in rows if r["arm"] == arm]
            steps = sorted(set.intersection(*[{x["step"] for x in rs} for rs in runs]))
            curves[arm] = {s: tuple(st.mean(chan(next(x for x in rs if x["step"] == s), mode, k)
                                            for rs in runs) for k in ("hack_freq", "retain"))
                           for s in steps}
        print(f"\n-- {label} ({mode}) --")
        print(f"{'lr/3 step':>9s} {'hack':>6s} {'retain':>6s} | {'lr 5e-4 step':>12s} {'hack':>6s} {'retain':>6s}")
        c1 = curves[a1]
        s1 = sorted(c1)
        for s3, (h3, r3) in sorted(curves[a3].items()):
            t = step_at_cum(CUM[s3] / 3)
            right = f"{t:12.1f}  (before its first eval)"
            if s1[0] <= t <= s1[-1]:
                lo = max(s for s in s1 if s <= t)
                hi = min(s for s in s1 if s >= t)
                w = 0.0 if hi == lo else (t - lo) / (hi - lo)
                h1, r1 = ((1 - w) * c1[lo][i] + w * c1[hi][i] for i in (0, 1))
                right = f"{t:12.1f} {h1:6.3f} {r1:6.3f}"
            print(f"{s3:9d} {h3:6.3f} {r3:6.3f} | {right}")
        h1, r1 = c1[s1[-1]]
        print(f"{'':9s} {'':6s} {'':6s} | {s1[-1]:12d} {h1:6.3f} {r1:6.3f}  (lr 5e-4 run, last eval)")

if __name__ == "__main__":
    main()
