"""Stage and upload Countdown adapters (repro_data/runs.csv) to a Hugging Face repo,
one folder per run: <class>/<run_name>/ with

  model.safetensors, dual_lora_config.json   checkpoint-200 (the adapter weights only)
  run_config.yaml                            the training config
  eval/fseval.json, eval/probe.json          n=256 fseval and expr-only probe (k=1)
  meta.json                                  arm, sweep, code commit, base model, lr, seed,
                                             deploy scale, sha256 of model.safetensors

plus a root README.md indexing every run. Meant to run where the checkpoints are
local (the runs' rl/output/, or wherever it was synced to); uploads in one commit
with HfApi.upload_folder (files already on the Hub are skipped). Prints "<sha256>  <sweep>/<run>/
checkpoint-200/model.safetensors" for every run (for repro_data/MANIFEST.txt).

    python tools/hf_upload_adapters.py --runs repro_data/runs.csv --output_root output \
        --stage <dir> --repo samuel500/graft-repro-adapters
    # README + metadata only, weights already on the Hub (from rl/; upload_folder adds or
    # replaces the staged files and leaves the weights alone):
    python tools/hf_upload_adapters.py --runs repro_data/runs.csv --output_root repro_data \
        --stage <dir> --no_ckpt --manifest repro_data/MANIFEST.txt
    # the loreft branch's runs (LoReFT, MLP + LoReFT, no-anchor) to their own repo:
    python tools/hf_upload_adapters.py --runs repro_data/runs.csv --output_root output \
        --stage <dir> --collection loreft --repo samuel500/graft-loreft-adapters
"""
import argparse
import csv
import hashlib
import json
import os
import shutil

BASE_REPO = "samuel500/graft-qwen3-8b-countdown-sft"
BASE_REV = "a2d54fdf0aa9fa6ee8e487777ee7d63adaf8ec48"
CODE_REPO = "https://github.com/samuel500/GRAFT"  # fork of jnward/GRAFT
PAPER = "https://arxiv.org/abs/2610.17062"
REPRO_CLASS_DOC = {
    "graft": "GRAFT: dual MLP adapters (64 RETAIN + 64 FORGET neurons per layer), "
             "detector-flagged hacks routed to FORGET. Deploy with FORGET ablated (scale 0.0).",
    "graft-no-routing": "The paper's GRAFT w/o routing control: same dual adapters and anchor as "
                        "GRAFT, detector recall 0 so nothing is routed. Reported at scale 0.0 "
                        "like GRAFT; scale 1.0 = both adapters.",
    "no-intervention": "Standard RL, nothing ablated (scale 1.0). Paper config: dual m64 adapters "
                       "with routing off. The +anchor arms train ONE 128-neuron adapter "
                       "(forget_neurons 0) on the GRAFT recipe's LeetCode anchor.",
}
# One collection per HF repo: the runs.csv sweeps it holds, the code branch its README and
# meta.json point to, its title / intro / class docs, and the run the loading example uses.
COLLECTIONS = {
    "repro": dict(
        sweeps={"repro_fig2", "dn_anchor", "dn_anchor_lr1", "dn_lr3", "dn_paper_s1", "gr_lr1",
                "noroute_lr1"},
        branch="repro",
        title="GRAFT Countdown repro adapters",
        intro=("Adapters from our reproduction of the Countdown-Code experiments (Fig 2) of the "
               f"[GRAFT paper]({PAPER}), plus follow-up arms (no-intervention anchor x learning "
               "rate, GRAFT and GRAFT w/o routing at lr 5e-4)."),
        class_doc=REPRO_CLASS_DOC,
        doc="REPRO.md",
        example="graft/cdhf100_gr_lccoh64_lr3_s9",
        scale_note=("**0.0** = RETAIN only (GRAFT's deployed model; also how the paper reports "
                    "GRAFT w/o routing), **1.0** = both adapters (No intervention, which has no "
                    "FORGET adapter to ablate)."),
        loading_note=", including the single 128-neuron adapter of the +anchor no-intervention arms",
        vllm=("For vLLM serving (`vllm_utils.py`), pass the run's `mlp_config` from "
              "`run_config.yaml`: `m64` for GRAFT, GRAFT w/o routing and the paper-config "
              "no-intervention runs, `m128_retain_only` for the +anchor no-intervention runs."),
    ),
    "loreft": dict(
        sweeps={"exp_loreft", "exp_m64_loreft_local", "exp_nocoh"},
        branch="loreft",
        title="GRAFT Countdown LoReFT adapters",
        intro=("Follow-ups to our reproduction of the Countdown-Code experiments of the "
               f"[GRAFT paper]({PAPER}): GRAFT with LoReFT adapters (Wu et al. 2024) on one or "
               "both sides, down to a rank-1 LoReFT FORGET adapter on a single layer next to the "
               "paper's MLP RETAIN adapter, and GRAFT without the LeetCode anchor. The paper-config "
               "MLP GRAFT adapters (with the anchor) are in "
               "[`samuel500/graft-repro-adapters`](https://huggingface.co/samuel500/graft-repro-adapters)."),
        class_doc={
            "loreft": "LoReFT on both sides (`loreft:<rank>` RETAIN and FORGET, all layers or "
                      "layer 20), classic routing, LeetCode anchor. Deploy at scale 0.0.",
            "mlp-loreft": "The paper's MLP RETAIN adapter (64 neurons, every layer) with a rank-1 "
                          "LoReFT FORGET adapter on a few layers (`forget_layers` in "
                          "`run_config.yaml`); κ_F in `adapter_config`. Deploy at scale 0.0. "
                          "`anchor` is `none` for the runs without the LeetCode anchor.",
            "graft": "Paper-config MLP GRAFT (dual m64) trained WITHOUT the LeetCode anchor. "
                     "Deploy at scale 0.0.",
        },
        doc="LOREFT.md",
        example="mlp-loreft/cdhf100_gr_m64_lf1_L20_k8_s9",
        scale_note=("**0.0** = RETAIN only, the deployed model, for every run here; **1.0** = "
                    "both adapters (the `both: hack` column of LOREFT.md)."),
        loading_note=", including composite MLP / LoReFT plans",
        vllm=("For vLLM serving, composite (LoReFT) plans need eager mode "
              "(`vllm_mlp_adapter.VLLMDualLoReFT`); `run_config.yaml` has the plan "
              "(`retain_adapter`, `forget_adapter`, `*_layers`)."),
        extra=[
            "", "## Steering vector", "",
            "`steering_vector/` holds the paper's preventative-steering (PPS) baseline vector, "
            "re-extracted by us on the SFT base with the paper code's recipe "
            "(`tools/modal_pps_extract.py`, extract step: 180 hacking vs 418 clean rollouts on "
            "hackable questions; the paper's own vector file was not published). It is used in "
            f"[`experiments/steering_variance`]({CODE_REPO}/blob/loreft/experiments/"
            "steering_variance/README.md), which compares it with the layer-20 LoReFT FORGET "
            "directions of the `mlp-loreft/*_L20_k8_*` runs.", "",
            "- `pps_vector_hackonly.pt` (`torch.load`): `{\"raw\": {L: fp32[4096]}, "
            "\"incremental\": {L: fp32[4096]}, \"meta\": {...}}`. `raw[L]` is the hacking minus "
            "clean mean of the response-token residual stream at the output of decoder layer `L` "
            "(0-based). The paper's PPS arm adds `2 * raw[20]` at layer 20 during training. On "
            "the SFT base with a neutral prompt that raises test_overwrite from 3.7% to 25.6% "
            "and lowers true solves from 52% to 25% (1,024 rollouts each).",
            "- `natural_h20.npz`: `hack` [553, 4096] and `clean` [1399, 4096], the response-mean "
            "layer-20 residual (SFT-base replay) of natural rollouts from the SFT base and the "
            "no-intervention run `cdhf100_dn_lr3_s9`; `dir_dn` [4096], the natural hack "
            "direction. `loreft_cosine.py --hf` uses them to orient the LoReFT directions.",
        ],
    ),
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


def stage_run(r, out, stage, with_ckpt, ck_sha, branch):
    src = f"{out}/{r['sweep_name']}/{r['run_name']}"
    dst = f"{stage}/{r['hf_path']}"
    ck = f"{src}/checkpoint-200"
    meta = {k: r[k] for k in ("arm", "class", "run_name", "seed", "lr", "anchor", "routing",
                              "adapter_config", "deploy_scale", "code_commit", "finished")}
    meta["seed"] = int(meta["seed"])
    meta.update(sweep_file=r["sweep_file"], sweep_name=r["sweep_name"], code_repo=CODE_REPO,
                code_branch=branch, base_model=BASE_REPO, base_model_revision=BASE_REV, step=200)
    if with_ckpt:
        for f in ("model.safetensors", "dual_lora_config.json"):
            put(f"{ck}/{f}", f"{dst}/{f}")
        meta["model_safetensors_sha256"] = sha256(f"{ck}/model.safetensors")
    elif ck_sha:
        meta["model_safetensors_sha256"] = ck_sha[f"{r['sweep_name']}/{r['run_name']}"]
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


def readme(rows, results, repo, col):
    b, d = col["branch"], col["doc"]
    lines = [
        "---", "license: apache-2.0", f"base_model: {BASE_REPO}", "tags: [graft, gradient-routing, "
        "reward-hacking, countdown]", "---", "",
        f"# {col['title']}", "",
        f"{col['intro']} All runs train adapters on the frozen SFT base "
        f"[`{BASE_REPO}`](https://huggingface.co/{BASE_REPO}) (revision `{BASE_REV[:8]}`) for 200 "
        "steps on 1x H200; the base model's card lists its SFT data source and recipe. The "
        f"experiments, rerun commands and caveats are documented in [`rl/{d}`]({CODE_REPO}"
        f"/blob/{b}/rl/{d}) on branch `{b}` of [{CODE_REPO[19:]}]({CODE_REPO}), a fork "
        "of the paper's code ([jnward/GRAFT](https://github.com/jnward/GRAFT)); "
        f"[`rl/repro_data/runs.csv`]({CODE_REPO}/blob/{b}/rl/repro_data/runs.csv) there is the "
        "same index as below.", "",
        "## Layout", "",
        "`<class>/<run_name>/`: `model.safetensors` + `dual_lora_config.json` (checkpoint-200, "
        "adapter weights only), `run_config.yaml`, `eval/fseval.json` (n=256 fseval), "
        "`eval/probe.json` (expr-only pass@1, k=1, n=256), `meta.json` (code commit, sweep, "
        "base model, lr, seed, deploy scale, sha256).", "",
    ]
    for c, doc in col["class_doc"].items():
        lines.append(f"- `{c}/`: {doc}")
    lines += col.get("extra", [])
    lines += [
        "", "## Which scale to deploy", "",
        "Scales are (retain, forget) multipliers on the two adapters. `deploy_scale` is the "
        f"forget scale with retain at 1: {col['scale_note']}", "",
        "## Loading", "",
        "With the code repo's `rl/` on `sys.path` (HF transformers path; the adapter shape is "
        f"read from `dual_lora_config.json`{col['loading_note']}):", "",
        "```python",
        "from huggingface_hub import snapshot_download",
        "from eval_utils import load_gradient_routing_model",
        "from gradient_routing import set_scales",
        "",
        f'base = snapshot_download("{BASE_REPO}", revision="{BASE_REV}")',
        f'ckpt = snapshot_download("{repo}",',
        f'                          allow_patterns="{col["example"]}/*")',
        f'model = load_gradient_routing_model(f"{{ckpt}}/{col["example"]}", base_model=base)',
        "set_scales(model, retain_scale=1.0, forget_scale=0.0)  # deploy: FORGET ablated",
        "```", "",
        col["vllm"], "",
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
    ap.add_argument("--no_ckpt", action="store_true", help="skip weights (README/meta only)")
    ap.add_argument("--manifest", help="with --no_ckpt: take checkpoint sha256s for meta.json "
                    "from repro_data/MANIFEST.txt")
    ap.add_argument("--no_upload", action="store_true")
    ap.add_argument("--collection", choices=sorted(COLLECTIONS), default="repro",
                    help="which runs.csv rows, README text and code branch")
    args = ap.parse_args()
    col = COLLECTIONS[args.collection]
    rows = [r for r in csv.DictReader(open(args.runs))
            if r["sweep_name"] in col["sweeps"]]
    ck_sha = {}
    if args.manifest:
        for line in open(args.manifest):
            sha, path = line.split()
            if path.endswith("/checkpoint-200/model.safetensors"):
                ck_sha[path.rsplit("/", 2)[0]] = sha
    results = {}
    for r in rows:
        meta, results[r["run_name"]] = stage_run(r, args.output_root, args.stage,
                                                 not args.no_ckpt, ck_sha, col["branch"])
        if not args.no_ckpt:
            print(f"{meta['model_safetensors_sha256']}  {r['sweep_name']}/{r['run_name']}/"
                  "checkpoint-200/model.safetensors", flush=True)
    with open(f"{args.stage}/README.md", "w") as f:
        f.write(readme(rows, results, args.repo, col))
    print(f"staged {len(rows)} runs in {args.stage}")
    if not args.no_upload:
        from huggingface_hub import HfApi
        HfApi().upload_folder(repo_id=args.repo, repo_type="model", folder_path=args.stage,
                              commit_message=(f"{len(rows)} countdown {args.collection} adapters"
                                              + (" (README + metadata)" if args.no_ckpt else "")))
        print("UPLOAD_OK")


if __name__ == "__main__":
    main()
