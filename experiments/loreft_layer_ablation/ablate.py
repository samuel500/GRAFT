"""Layer-window ablations of a trained composite LoReFT checkpoint (retain and
forget both LoReFT on every layer, e.g. the exp_loreft r1 screen arms).

Runs the post-training battery (fseval n=256 + expr-only probe k=1 n=256, as in
rl/scripts/pod/run_exp.sh) once per ablation config, to locate where along the
depth the task gain (retain side) and the hacking (forget side) sit.

Part A (retain): forget off (scale 0); retain LoReFT active only on a subset.
  retain_all, retain_none, and for each window W: retain_keep_W (only W
  active), retain_drop_W (all but W active).
Part B (forget): retain on everywhere, forget at scale 1 on a subset.
  forget_all, and for each window W: forget_keep_W, forget_drop_W.
  (forget_none is retain_all.)

A dropped layer's intervention is neutralized in the weights: W := R and
b := 0, so A = W - R = 0 and the edit (h·Aᵀ + b)·R is exactly zero, whatever
scales the eval code sets. Checked per config before generating.

One JSON per config in <out>/<run_name>/configs/<config>.json (skipped when
present, so reruns resume); per-sample fseval values and probe completions in
<out>/<run_name>/per_sample/.

    python experiments/loreft_layer_ablation/ablate.py \
        --ckpt output/exp_loreft/cdhf100_gr_loreft_r1_lr3_s9/checkpoint-200 \
        --out output/loreft_layer_ablation [--shard 0 --num_shards 4] [--only retain_keep_1-5]
"""
import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "rl"))
sys.path.insert(0, os.path.join(_REPO, "rl", "tools"))

# Inclusive decoder-layer indices (Qwen3-8B: layers 0-35).
DEFAULT_WINDOWS = "1-5,6-10,11-15,16-20,21-25,26-30,31-35"


def parse_windows(spec):
    out = []
    for item in spec.split(","):
        a, b = (int(x) for x in item.split("-"))
        assert a <= b, f"bad window {item!r}"
        out.append((a, b))
    return out


def build_configs(n_layers, windows):
    every = list(range(n_layers))
    cfgs = [
        {"name": "retain_all", "part": "retain", "retain": every, "forget": [], "forget_scale": 0.0},
        {"name": "retain_none", "part": "retain", "retain": [], "forget": [], "forget_scale": 0.0},
    ]
    for a, b in windows:
        win = list(range(a, b + 1))
        rest = [i for i in every if i not in win]
        cfgs += [
            {"name": f"retain_keep_{a}-{b}", "part": "retain", "retain": win, "forget": [],
             "forget_scale": 0.0},
            {"name": f"retain_drop_{a}-{b}", "part": "retain", "retain": rest, "forget": [],
             "forget_scale": 0.0},
        ]
    cfgs.append({"name": "forget_all", "part": "forget", "retain": every, "forget": every,
                 "forget_scale": 1.0})
    for a, b in windows:
        win = list(range(a, b + 1))
        rest = [i for i in every if i not in win]
        cfgs += [
            {"name": f"forget_keep_{a}-{b}", "part": "forget", "retain": every, "forget": win,
             "forget_scale": 1.0},
            {"name": f"forget_drop_{a}-{b}", "part": "forget", "retain": every, "forget": rest,
             "forget_scale": 1.0},
        ]
    return cfgs


def loreft_modules(model):
    """{layer_idx: DualLoReFT} for every adapted decoder layer."""
    from gradient_routing import DualLoReFT
    mods = {i: getattr(layer, "loreft", None) for i, layer in enumerate(model.model.layers)}
    mods = {i: m for i, m in mods.items() if isinstance(m, DualLoReFT)}
    assert mods, "checkpoint has no DualLoReFT modules"
    return mods


def snapshot(mods):
    """Trained (W, b) per (layer, role), cloned."""
    out = {}
    for i, m in mods.items():
        for role in ("retain", "forget"):
            proj = getattr(m, f"proj_{role}")
            if proj is not None:
                out[(i, role)] = (proj.weight.detach().clone(), proj.bias.detach().clone())
    return out


def apply_config(mods, orig, cfg):
    """Restore every intervention, then neutralize those cfg leaves out."""
    import torch
    from gradient_routing import _orthonormal_rows
    with torch.no_grad():
        for (i, role), (w, b) in orig.items():
            proj = getattr(mods[i], f"proj_{role}")
            if i in cfg[role]:
                proj.weight.copy_(w)
                proj.bias.copy_(b)
            else:
                proj.weight.copy_(_orthonormal_rows(getattr(mods[i], f"rot_{role}")))
                proj.bias.zero_()
        for (i, role) in orig:
            A, _, b = mods[i].merged(role, torch.float32)
            zero = bool(A.abs().max() == 0 and b.abs().max() == 0)
            assert zero == (i not in cfg[role]), (
                f"{cfg['name']}: layer {i} {role} edit is {'zero' if zero else 'nonzero'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="output/loreft_layer_ablation")
    ap.add_argument("--windows", default=DEFAULT_WINDOWS)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--probe_k", type=int, default=1)
    ap.add_argument("--probe_batch", type=int, default=128)
    ap.add_argument("--gen_batch_size", type=int, default=None,
                    help="fseval prompts per generate call (default: all at once)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--only", default="", help="CSV of config names")
    ap.add_argument("--list", action="store_true", help="print the configs and exit")
    args = ap.parse_args()

    windows = parse_windows(args.windows)
    if args.list:
        for c in build_configs(36, windows):
            print(f"{c['name']:22s} fs={c['forget_scale']}  retain={len(c['retain'])}L  "
                  f"forget={len(c['forget'])}L")
        return

    import torch
    import yaml
    from transformers import AutoTokenizer, set_seed
    from eval_utils import (_find_run_config, load_gradient_routing_model,
                            posthoc_eval_from_checkpoint)
    from expr_only_probe import probe_rows, run_probe
    from gradient_routing import set_scales

    ckpt = args.ckpt.rstrip("/")
    run_name = os.path.basename(os.path.dirname(ckpt))
    rc = _find_run_config(ckpt) or os.path.join(os.path.dirname(ckpt), "run_config.yaml")
    run_cfg = yaml.safe_load(open(rc)) or {}
    base = run_cfg["model"]
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_gradient_routing_model(ckpt, base_model=base)
    model.to(torch.bfloat16).cuda().eval()
    mods = loreft_modules(model)
    orig = snapshot(mods)
    n_layers = len(model.model.layers)

    cfgs = build_configs(n_layers, windows)
    only = [s for s in args.only.split(",") if s]
    if only:
        unknown = set(only) - {c["name"] for c in cfgs}
        assert not unknown, f"unknown configs: {sorted(unknown)}"
        cfgs = [c for c in cfgs if c["name"] in only]
    cfgs = cfgs[args.shard::args.num_shards]

    cfg_dir = os.path.join(args.out, run_name, "configs")
    sample_dir = os.path.join(args.out, run_name, "per_sample")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    rows = probe_rows(args.n)
    print(f"[ablate {run_name}] shard {args.shard}/{args.num_shards}: "
          f"{[c['name'] for c in cfgs]}", flush=True)

    for cfg in cfgs:
        path = os.path.join(cfg_dir, f"{cfg['name']}.json")
        if os.path.exists(path):
            print(f"[ablate] skip {cfg['name']} (done)", flush=True)
            continue
        t0 = time.time()
        apply_config(mods, orig, cfg)
        fs = cfg["forget_scale"]

        res = posthoc_eval_from_checkpoint(
            model, tok, ckpt, n_eval=args.n, modes=[(cfg["name"], 1.0, fs)],
            run_config_path=rc, gen_batch_size=args.gen_batch_size)[cfg["name"]]
        fseval = {k: v["mean"] for k, v in res.get("metrics", {}).items()
                  if isinstance(v, dict) and v.get("mean") is not None}
        # Per-sample metric values (for CIs); eval results keep no full completions.
        with open(os.path.join(sample_dir, f"{cfg['name']}__fseval_values.json"), "w") as f:
            json.dump({k: v["values"] for k, v in res.get("metrics", {}).items()
                       if isinstance(v, dict) and "values" in v}, f, default=float)
        t_fs = time.time() - t0

        set_scales(model, 1.0, fs)
        tok.padding_side = "left"
        set_seed(42)
        probe, recs = run_probe(model, tok, rows, k=args.probe_k, batch_prompts=args.probe_batch,
                                label=f"{run_name} {cfg['name']}")
        set_scales(model, 1.0, 1.0)
        with open(os.path.join(sample_dir, f"{cfg['name']}__probe.jsonl"), "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")

        out = {"run_name": run_name, "ckpt": ckpt, "config": cfg, "n": args.n,
               "probe_k": args.probe_k, "fseval": fseval, "probe": probe,
               "elapsed_s": {"fseval": round(t_fs), "total": round(time.time() - t0)}}
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[ablate] {cfg['name']} done in {time.time() - t0:.0f}s: "
              f"hack={fseval.get('hack_freq/score_threshold')} "
              f"pass@1={probe['true_pass1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
