"""Cosine similarity between the PPS steering vector (hackonly v_20) and the
write directions of the 8 kappa_F=8 layer-20 LoReFT forget adapters
(cdhf100_gr_m64_lf1_L20_k8_s{1,2,3,4,5,9,15,16}: MLP retain, rank-1 LoReFT
forget on the output of decoder layer 20, the point PPS adds to).

A rank-1 LoReFT edits h -> h + c(h) R with c(h) = (W - R) h + b, R the unit
row (rot_forget normalized, as _orthonormal_rows does for rank 1). R's sign is
arbitrary, so each R is oriented by the sign of its mean edit coefficient over
the natural rollouts (c is linear, so the mean over a rollout's tokens is
c(response-mean h_20); the means come from the SFT-base replay in
capture/base/proj.npz, which approximates the LoReFT model's own layer-20
input). The oriented R is the direction the adapter pushes on average.

Inputs, local (results/data/): loreft/<run>/model.safetensors, pps/pps_vector_hackonly.pt,
capture/base/{proj.npz,index.json}, rollouts_{base,dn}.json, analysis/proj_nat_base.npz.
Inputs, --hf: the same from Hugging Face (HF_REPO at HF_REV): mlp-loreft/<run>/model.safetensors,
steering_vector/pps_vector_hackonly.pt, steering_vector/natural_h20.npz (the per-rollout
response-mean h_20 of both classes and the natural direction; written by --export).
Outputs: results/plots/loreft_cosine.{png,json} (or --out)

    uv run --no-project --with torch --with safetensors --with numpy --with matplotlib \
        --with huggingface_hub --index https://download.pytorch.org/whl/cpu \
        python experiments/steering_variance/loreft_cosine.py --hf
    # local data instead: drop --hf; --export <dir> stages steering_vector/ for the HF repo
"""
import argparse
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from safetensors import safe_open

from plot import CLEAN, DATA, GRID, HERE, LAYER, SURFACE, TEXT, TEXT_2

SEEDS = (1, 2, 3, 4, 5, 9, 15, 16)
RUN = "cdhf100_gr_m64_lf1_L20_k8_s{}"
PREFIX = f"_orig_mod.model.layers.{LAYER}.loreft."
HF_REPO = "samuel500/graft-loreft-adapters"
HF_REV = "4eaf2b1c7a93c0d4f8ecfc91b0e9bd8286903423"
HF_DIR = "steering_vector"
# Diverging pair (dataviz reference palette): blue <-> red, gray midpoint.
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])


def pps_vector(path):
    import torch
    d = torch.load(path, map_location="cpu")
    return d["raw"][LAYER].numpy().astype(np.float64)


def loreft(path):
    f = safe_open(path, "np")
    rot = f.get_tensor(PREFIX + "rot_forget").astype(np.float64)[0]
    W = f.get_tensor(PREFIX + "proj_forget.weight").astype(np.float64)[0]
    b = float(f.get_tensor(PREFIX + "proj_forget.bias")[0])
    R = rot / np.linalg.norm(rot)
    return R, W - R, b


def natural_local():
    """Response-mean h_20 (SFT-base replay) of the natural rollouts, by class, and
    the natural (no-intervention) hack direction."""
    index = json.load(open(os.path.join(DATA, "capture", "base", "index.json")))["rollouts"]
    mean20 = np.load(os.path.join(DATA, "capture", "base", "proj.npz"))["mean20"]
    src = {s: [x for x in json.load(open(os.path.join(DATA, f"rollouts_{s}.json")))["rollouts"]
               if x["response_token_ids"]] for s in ("base", "dn")}
    out = {"hack": [], "clean": []}
    for k, r in enumerate(index):
        if r["source"] not in src:
            continue
        x = src[r["source"]][r["i"]]
        if x["test_overwrite"]:
            out["hack"].append(mean20[k])
        elif x["test_modified"] is False:
            out["clean"].append(mean20[k])
    out = {c: np.array(v) for c, v in out.items()}
    out["dir_dn"] = np.load(os.path.join(DATA, "analysis", "proj_nat_base.npz"))["dir_dn"]
    return out


def inputs_local():
    return (os.path.join(DATA, "pps", "pps_vector_hackonly.pt"),
            {s: os.path.join(DATA, "loreft", RUN.format(s), "model.safetensors") for s in SEEDS},
            natural_local())


def inputs_hf(rev):
    from huggingface_hub import hf_hub_download

    def get(f):
        return hf_hub_download(HF_REPO, f, revision=rev)
    nat = np.load(get(f"{HF_DIR}/natural_h20.npz"))
    return (get(f"{HF_DIR}/pps_vector_hackonly.pt"),
            {s: get(f"mlp-loreft/{RUN.format(s)}/model.safetensors") for s in SEEDS},
            {k: nat[k] for k in ("hack", "clean", "dir_dn")})


def export(out_dir):
    """Stage the HF repo's steering_vector/ from local data."""
    os.makedirs(out_dir, exist_ok=True)
    vec, _, nat = inputs_local()
    shutil.copy(vec, os.path.join(out_dir, "pps_vector_hackonly.pt"))
    np.savez(os.path.join(out_dir, "natural_h20.npz"), **nat)
    print(f"{out_dir}: pps_vector_hackonly.pt, natural_h20.npz "
          f"(hack {len(nat['hack'])}, clean {len(nat['clean'])} rollouts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="store_true", help=f"inputs from {HF_REPO}")
    ap.add_argument("--rev", default=HF_REV, help="--hf: repo revision")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "plots"))
    ap.add_argument("--export", metavar="DIR", help="only stage steering_vector/ for the HF repo")
    args = ap.parse_args()
    if args.export:
        return export(args.export)
    vec_path, loreft_paths, nat = inputs_hf(args.rev) if args.hf else inputs_local()
    v = pps_vector(vec_path)
    u_pps = v / np.linalg.norm(v)
    H = {c: nat[c].astype(np.float64) for c in ("hack", "clean")}
    u_nat = nat["dir_dn"].astype(np.float64)
    u_nat = u_nat / np.linalg.norm(u_nat)
    names, units, info = ["PPS $v_{20}$"], [u_pps], {}
    for s in SEEDS:
        R, A, b = loreft(loreft_paths[s])
        c = {cls: H[cls] @ A + b for cls in H}           # mean edit coefficient per rollout
        sign = np.sign(np.concatenate(list(c.values())).mean()) or 1.0
        units.append(sign * R)
        names.append(f"LoReFT s{s}")
        info[f"s{s}"] = {
            "orientation_sign": float(sign),
            "coef_mean_hack": float(sign * c["hack"].mean()),
            "coef_mean_clean": float(sign * c["clean"].mean()),
            "frac_rollouts_coef_positive": float(np.mean(np.concatenate(list(c.values())) * sign > 0)),
            "cos_pps": float(sign * R @ u_pps),
            "cos_natural_dn": float(sign * R @ u_nat),
            "norm_W_minus_R": float(np.linalg.norm(A)), "bias": b,
        }
    U = np.stack(units)
    C = U @ U.T
    off = C[1:, 1:][~np.eye(len(SEEDS), dtype=bool)]
    summary = {"names": [n.replace("$v_{20}$", "v20") for n in names], "cosine": C.tolist(),
               "loreft_vs_pps": {"mean": float(C[0, 1:].mean()), "abs_mean": float(np.abs(C[0, 1:]).mean()),
                                 "min": float(C[0, 1:].min()), "max": float(C[0, 1:].max())},
               "loreft_vs_loreft": {"mean": float(off.mean()), "abs_mean": float(np.abs(off).mean()),
                                    "min": float(off.min()), "max": float(off.max())},
               "cos_pps_natural_dn": float(u_pps @ u_nat),
               "random_unit_pair_std_4096": 1 / np.sqrt(4096),
               "per_seed": info}
    out = args.out
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "loreft_cosine.json"), "w") as f:
        json.dump(summary, f, indent=1)

    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
                         "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    im = ax.imshow(C, cmap=DIV, vmin=-1, vmax=1)
    n = len(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color=SURFACE if abs(C[i, j]) > 0.6 else TEXT)
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(False, which="major")
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    ax.axhline(0.5, color=TEXT_2, lw=1.2)
    ax.axvline(0.5, color=TEXT_2, lw=1.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("cosine similarity", color=TEXT_2)
    cb.outline.set_edgecolor(GRID)
    ax.set_title("Layer 20: PPS steering vector vs the 8 κ$_F$=8 LoReFT forget directions\n"
                 "(each LoReFT direction oriented by its mean edit on natural rollouts)",
                 loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "loreft_cosine.png"), dpi=150)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("cosine", "names")}, indent=1))
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    print(C)


if __name__ == "__main__":
    main()
