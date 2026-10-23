"""Tabulate ablate.py results: one markdown table per part, with each window's
keep-only and drop-only configs side by side, written to
<results_dir>/<run_name>/summary.md and printed.

    python experiments/loreft_layer_ablation/analyze.py \
        experiments/loreft_layer_ablation/results/cdhf100_gr_loreft_r1_lr3_s9

With --aggregate OUT and several run dirs (seeds), writes OUT/summary.md with
the mean ± SEM across runs per config, plus each run's reference configs.

    python experiments/loreft_layer_ablation/analyze.py --aggregate results/<group> \
        results/<run_s1> results/<run_s9> ...
"""
import glob
import json
import math
import os
import sys

# Short name -> fseval metric key prefix (exact first path component).
FSEVAL = {"comb": "combined", "retain": "retain", "hack": "hack_freq"}


def load(run_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "configs", "*.json"))):
        r = json.load(open(p))
        row = {"probe": r["probe"]["true_pass1"], "parse": r["probe"]["parse_rate"]}
        for short, head in FSEVAL.items():
            keys = [k for k in r["fseval"] if k.split("/")[0] == head]
            assert len(keys) == 1, f"{p}: {head} keys {keys}"
            row[short] = r["fseval"][keys[0]]
        out[r["config"]["name"]] = row
    return out


KEYS = ("comb", "retain", "hack", "probe", "parse")


def aggregate(rowsets):
    """Per config and metric: mean over runs, and SEM across runs under '<key>_e'."""
    names = set(rowsets[0])
    assert all(set(r) == names for r in rowsets), "runs have different configs"
    out = {}
    for name in names:
        row = {}
        for k in KEYS:
            xs = [r[name][k] for r in rowsets]
            m = sum(xs) / len(xs)
            row[k] = m
            row[k + "_e"] = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))
        out[name] = row
    return out


def fmt(row, ref=None):
    cells = []
    for k in KEYS:
        if row is None:
            cells.append("")
            continue
        c = f"{row[k]:.3f}"
        if k + "_e" in row:
            c += f" ± {row[k + '_e']:.3f}"
        if ref is not None:
            c += f" ({row[k] - ref[k]:+.3f})"
        cells.append(c)
    return cells


def table(rows, part, windows, refs):
    head = ["config", "comb", "retain", "hack", "probe pass@1", "probe parse"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name in refs:
        if name in rows:
            lines.append("| " + " | ".join([f"**{name}**"] + fmt(rows[name])) + " |")
    keep_ref, drop_ref = refs
    for w in windows:
        for kind, ref in (("keep", keep_ref), ("drop", drop_ref)):
            name = f"{part}_{kind}_{w}"
            if name in rows:
                lines.append("| " + " | ".join([name] + fmt(rows[name], rows.get(ref))) + " |")
    return "\n".join(lines)


def main_aggregate(dest, run_dirs):
    rowsets = [load(d) for d in run_dirs]
    rows = aggregate(rowsets)
    windows = sorted({n.split("_", 2)[2] for n in rows if "_keep_" in n},
                     key=lambda w: int(w.split("-")[0]))
    runs = [os.path.basename(d) for d in run_dirs]
    head = ["run", "config", "comb", "retain", "hack", "probe pass@1", "probe parse"]
    per_run = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for run, r in zip(runs, rowsets):
        for name in ("retain_none", "retain_all", "forget_all"):
            per_run.append("| " + " | ".join([run, name] + fmt(r[name])) + " |")
    md = [
        f"# Layer-window ablations: mean of {len(runs)} runs",
        "",
        "Runs: " + ", ".join(f"`{r}`" for r in runs) + ". Per-run tables and plots are in",
        "`../<run>/`. Each cell is the mean over runs ± the SEM across runs (n=256",
        "per run); keep_W deltas are vs the all-off reference, drop_W vs the all-on",
        "one, both on the means. Plots (mean ± SEM; effect shares are the mean of",
        "the per-run shares) are in `plots/`.",
        "",
        "## References per run",
        "",
        "\n".join(per_run),
        "",
        "## Part A: retain adapter windows, forget off (scale 0)",
        "",
        table(rows, "retain", windows, ("retain_none", "retain_all")),
        "",
        "## Part B: forget adapter windows at scale 1, retain on everywhere",
        "",
        table(rows, "forget", windows, ("retain_all", "forget_all")),
        "",
    ]
    text = "\n".join(md)
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "summary.md"), "w") as f:
        f.write(text)
    print(text)


def main():
    if sys.argv[1] == "--aggregate":
        return main_aggregate(sys.argv[2].rstrip("/"), [d.rstrip("/") for d in sys.argv[3:]])
    run_dir = sys.argv[1].rstrip("/")
    rows = load(run_dir)
    windows = sorted({n.split("_", 2)[2] for n in rows if "_keep_" in n or "_drop_" in n},
                     key=lambda w: int(w.split("-")[0]))
    md = [
        f"# Layer-window ablations: {os.path.basename(run_dir)}",
        "",
        "fseval n=256 (comb / retain reward / hack rate) and expr-only probe pass@1, k=1",
        "n=256. One SE at p=0.5 is about 0.031. keep_W rows show the delta vs the",
        "all-off reference, drop_W rows vs the all-on reference.",
        "",
        "## Folder layout",
        "",
        "- `summary.md`: this file, written by `analyze.py`.",
        "- `plots/`: charts from `plot.py`. `retain_windows.png` and `forget_windows.png`",
        "  show per-window metrics against the all-on and all-off references;",
        "  `effect_share.png` shows each window's share of the full effect.",
        "- `configs/<config>.json`: one file per config, the source of every number",
        "  here. `config` says which layers each adapter has on and the forget",
        "  scale. `fseval` has the fseval means (`retain/...` = solve rate,",
        "  `hack_freq/...` = hack rate, `combined/...` = reward by either route,",
        "  plus hack-type and detected/undetected splits). `probe` has the",
        "  expr-only probe stats (`true_pass1`, `parse_rate`). `elapsed_s` has timings.",
        "- `per_sample/<config>__fseval_values.json`: for every fseval metric, the",
        "  256 per-problem values (0/1, `null` where a slice doesn't apply).",
        "  Position i is the same problem in every config.",
        "- `per_sample/<config>__probe.jsonl`: one line per probe problem: `idx`,",
        "  `nums`, `target`, the full `completion`, the extracted `expr`, `src`",
        "  (expr json or the solution.py fallback) and `true` (correct or not).",
        "",
        "Config names are `<part>_<keep|drop>_<window>`. `keep` means only that",
        "window is on; `drop` means all but that window. The references are",
        "`retain_all`, `retain_none` and `forget_all`. Windows are 0-indexed",
        "decoder layers; layer 0 is never ablated. Full fseval completions were",
        "not saved (the eval code returns only scores), so per-problem hack text",
        "isn't available.",
        "",
        "## Part A: retain adapter windows, forget off (scale 0)",
        "",
        "keep deltas vs retain_none (the SFT model), drop deltas vs retain_all.",
        "",
        table(rows, "retain", windows, ("retain_none", "retain_all")),
        "",
        "## Part B: forget adapter windows at scale 1, retain on everywhere",
        "",
        "keep deltas vs retain_all (forget off), drop deltas vs forget_all.",
        "",
        table(rows, "forget", windows, ("retain_all", "forget_all")),
        "",
    ]
    text = "\n".join(md)
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
