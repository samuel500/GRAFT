"""Stage and upload the Countdown repro adapters (repro_data/runs.csv) to a Hugging
Face repo, one folder per run: <class>/<run_name>/ with

  model.safetensors, dual_lora_config.json   checkpoint-200 (the adapter weights only)
  run_config.yaml                            the training config
  eval/fseval.json, eval/probe.json          n=256 fseval and expr-only probe (k=1)
  meta.json                                  arm, sweep, code commit, base model, lr, seed,
                                             deploy scale, sha256 of model.safetensors

plus a root README.md indexing every run. Meant to run where the checkpoints are
local (a CPU pod with the network volume at /workspace); uploads in one commit
with HfApi.upload_folder (files already on the Hub are skipped). Prints "<sha256>  <sweep>/<run>/
checkpoint-200/model.safetensors" for every run (for repro_data/MANIFEST.txt).

    python hf_upload_adapters.py --runs runs.csv --output_root /workspace/graft-repro/output \
        --stage /workspace/graft-repro/hf_stage --repo samuel500/graft-repro-adapters
    # README only, no checkpoints (from rl/): --output_root repro_data --no_ckpt --no_upload
"""
import argparse
import csv
import hashlib
import json
import os
import shutil

BASE_REPO = "samuel500/graft-qwen3-8b-countdown-sft"
BASE_REV = "a2d54fdf0aa9fa6ee8e487777ee7d63adaf8ec48"
CODE_REPO = "https://github.com/samuel500/GRAFT"  # branch repro; public fork later
CLASS_DOC = {
    "graft": "GRAFT: dual MLP adapters (64 RETAIN + 64 FORGET neurons per layer), "
             "detector-flagged hacks routed to FORGET. Deploy with FORGET ablated (scale 0.0).",
    "graft-no-routing": "The paper's GRAFT w/o routing control: same dual adapters and anchor as "
                        "GRAFT, detector recall 0 so nothing is routed. Reported at scale 0.0 "
                        "like GRAFT; scale 1.0 = both adapters.",
    "no-intervention": "Standard RL, nothing ablated (scale 1.0). Paper config: dual m64 adapters "
                       "with routing off. The +anchor arms train ONE 128-neuron adapter "
                       "(forget_neurons 0) on the GRAFT recipe's LeetCode anchor.",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def put(src, dst):
    """Hard link where the filesystem allows (no 113 MB copies on the volume), else copy."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def val(d, p):
    hits = [v for k, v in d.items() if k.startswith(p) and "/" not in k[len(p):]]
    return hits[0] if hits else None


def stage_run(r, out, stage, with_ckpt):
    src = f"{out}/{r['sweep_name']}/{r['run_name']}"
    dst = f"{stage}/{r['hf_path']}"
    ck = f"{src}/checkpoint-200"
    meta = {k: r[k] for k in ("arm", "class", "run_name", "seed", "lr", "anchor", "routing",
                              "adapter_config", "deploy_scale", "code_commit", "finished")}
    meta["seed"] = int(meta["seed"])
    meta.update(sweep_file=r["sweep_file"], sweep_name=r["sweep_name"], code_repo=CODE_REPO,
                base_model=BASE_REPO, base_model_revision=BASE_REV, step=200)
    if with_ckpt:
        for f in ("model.safetensors", "dual_lora_config.json"):
            put(f"{ck}/{f}", f"{dst}/{f}")
        meta["model_safetensors_sha256"] = sha256(f"{ck}/model.safetensors")
    put(f"{src}/run_config.yaml", f"{dst}/run_config.yaml")
    fs = f"{out}/{r['sweep_name']}_fseval/{r['run_name']}.json"
    pr = f"{out}/expr_only_probe_arms/{r['run_name']}.json"
    put(fs, f"{dst}/eval/fseval.json")
    put(pr, f"{dst}/eval/probe.json")
    with open(f"{dst}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    sc = r["deploy_scale"]
    fsd, prd = json.load(open(fs))["scales"][sc], json.load(open(pr))["scales"][sc]
    return meta, (val(fsd, "hack_freq/"), prd["true_pass1"])


def readme(rows, results, repo):
    lines = [
        "---", "license: apache-2.0", f"base_model: {BASE_REPO}", "tags: [graft, gradient-routing, "
        "reward-hacking, countdown]", "---", "",
        "# GRAFT Countdown repro adapters", "",
        "Adapters from our reproduction of the Countdown-Code experiments (Fig 2) of the GRAFT "
        "paper, plus follow-up arms (no-intervention anchor x learning rate, GRAFT and GRAFT w/o "
        "routing at lr 5e-4). All runs train adapters on the frozen SFT base "
        f"[`{BASE_REPO}`](https://huggingface.co/{BASE_REPO}) (revision `{BASE_REV[:8]}`) for 200 "
        "steps on 1x H200. The experiments, rerun commands and caveats are documented in "
        f"`rl/REPRO.md` of the code repo ({CODE_REPO}, branch `repro`); `rl/repro_data/runs.csv` "
        "there is the same index as below.", "",
        "## Layout", "",
        "`<class>/<run_name>/`: `model.safetensors` + `dual_lora_config.json` (checkpoint-200, "
        "adapter weights only), `run_config.yaml`, `eval/fseval.json` (n=256 fseval), "
        "`eval/probe.json` (expr-only pass@1, k=1, n=256), `meta.json` (code commit, sweep, "
        "base model, lr, seed, deploy scale, sha256).", "",
    ]
    for c, doc in CLASS_DOC.items():
        lines.append(f"- `{c}/`: {doc}")
    lines += [
        "", "## Which scale to deploy", "",
        "Scales are (retain, forget) multipliers on the two adapters. `deploy_scale` is the "
        "forget scale with retain at 1: **0.0** = RETAIN only (GRAFT's deployed model; also how "
        "the paper reports GRAFT w/o routing), **1.0** = both adapters (No intervention, which "
        "has no FORGET adapter to ablate).", "",
        "## Loading", "",
        "With the code repo's `rl/` on `sys.path` (HF transformers path; the adapter shape is "
        "read from `dual_lora_config.json`, including the single 128-neuron adapter of the "
        "+anchor no-intervention arms):", "",
        "```python",
        "from huggingface_hub import snapshot_download",
        "from eval_utils import load_gradient_routing_model",
        "from gradient_routing import set_scales",
        "",
        f'base = snapshot_download("{BASE_REPO}", revision="{BASE_REV}")',
        f'ckpt = snapshot_download("{repo}",',
        '                          allow_patterns="graft/cdhf100_gr_lccoh64_lr3_s9/*")',
        'model = load_gradient_routing_model(f"{ckpt}/graft/cdhf100_gr_lccoh64_lr3_s9", base_model=base)',
        "set_scales(model, retain_scale=1.0, forget_scale=0.0)  # deploy: FORGET ablated",
        "```", "",
        "For vLLM serving (`vllm_utils.py`), pass the run's `mlp_config` from `run_config.yaml`: "
        "`m64` for GRAFT, GRAFT w/o routing and the paper-config no-intervention runs, "
        "`m128_retain_only` for the +anchor no-intervention runs.", "",
        "## Runs", "",
        "hack = ground-truth hack rate, pass@1 = expr-only pass@1, both at the deploy scale "
        "(n=256, from `eval/`). lr/3 = 5e-4/3, the paper's GRAFT lr.", "",
        "| class | run | arm | seed | lr | anchor | deploy scale | hack | pass@1 | code commit |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        h, p = results[r["run_name"]]
        link = f"https://huggingface.co/{repo}/tree/main/{r['hf_path']}"
        lines.append(f"| {r['class']} | [`{r['run_name']}`]({link}) | {r['arm']} | "
                     f"{r['seed']} | {r['lr']} | {r['anchor']} | {r['deploy_scale']} | {h:.3f} | "
                     f"{p:.3f} | `{r['code_commit']}` |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="repro_data/runs.csv")
    ap.add_argument("--output_root", required=True, help="dir with <sweep>/<run>/checkpoint-200")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--repo", default="samuel500/graft-repro-adapters")
    ap.add_argument("--no_ckpt", action="store_true", help="skip weights (README/meta test)")
    ap.add_argument("--no_upload", action="store_true")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.runs)))
    results = {}
    for r in rows:
        meta, results[r["run_name"]] = stage_run(r, args.output_root, args.stage, not args.no_ckpt)
        if "model_safetensors_sha256" in meta:
            print(f"{meta['model_safetensors_sha256']}  {r['sweep_name']}/{r['run_name']}/"
                  "checkpoint-200/model.safetensors", flush=True)
    with open(f"{args.stage}/README.md", "w") as f:
        f.write(readme(rows, results, args.repo))
    print(f"staged {len(rows)} runs in {args.stage}")
    if not args.no_upload:
        from huggingface_hub import HfApi
        HfApi().upload_folder(repo_id=args.repo, repo_type="model", folder_path=args.stage,
                              commit_message=f"{len(rows)} countdown repro adapters")
        print("UPLOAD_OK")


if __name__ == "__main__":
    main()
