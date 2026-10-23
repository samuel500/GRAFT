"""Per-token layer-20 activations along the preventative-steering (PPS) direction,
for natural (unprompted) hacking and non-hacking Countdown-Code rollouts.

Question: PPS adds one fixed vector (alpha * v_L, paper: hackonly, L20, alpha 2)
at every token. How does that constant shift compare with how realistic
activations actually move along the same direction, token by token, in hacking
vs non-hacking rollouts?

Phases (each its own process: vLLM and the HF models don't share a GPU well):

  pps        tools/modal_pps_extract.py::extract without Modal (same constants,
             read from that file with ast; same rollouts, grader filter,
             teacher-forced capture and disk format). Selection is skipped: the
             paper already fixed (hackonly, L20, alpha 2).
             -> <out>/pps/{rollouts.json,pps_vector_allpairs.pt,pps_vector_hackonly.pt}
  rollouts   natural rollouts under the NEUTRAL prompt on all-hackable eval-split
             questions (shuffle(42)[16000:], as the IP pre-screen), K per
             question, temperature 1 / top_p 1 / no top_k (the training policy's
             sampling). --source base: the SFT base (vLLM). --source dn: a
             no-intervention RL checkpoint at deploy scales (1, 1) (HF generate;
             the MLP adapters don't merge). Graded with countdown_all_components,
             plus test.py-modified flag, segment char offsets and per-token char
             ends so token positions can be aligned to the tamper point offline.
             -> <out>/rollouts_<source>.json
  capture    teacher-forced replay of every rollout (sources pps-kept, base, dn)
             through --model base (SFT) or dn (checkpoint, scales 1, 1). Hooks
             every decoder layer output (layer L = output of model.model.layers[L],
             the residual stream PPS adds to); response tokens only.
             -> <out>/capture/<model>/{acts_L20.npy (fp16 [n_tok, 4096], base and
                dn sources only), proj.npz, index.json}
  analyze    natural hack directions (cross-fitted difference of rollout means,
             test_overwrite vs test.py untouched, folds = question parity),
             projections of every token onto them, cosine table, per-class
             variance summary with the PPS shift alpha*||v_20|| for scale.
             -> <out>/analysis/{summary.json, proj_nat_<model>.npz}

    python experiments/steering_variance/steering_variance.py pps
    python experiments/steering_variance/steering_variance.py rollouts --source base
    python experiments/steering_variance/steering_variance.py rollouts --source dn --ckpt <dir>
    python experiments/steering_variance/steering_variance.py capture --model base
    python experiments/steering_variance/steering_variance.py capture --model dn --ckpt <dir>
    python experiments/steering_variance/steering_variance.py analyze
Add --smoke (and a separate --out) for a few-question end-to-end check.
"""
import argparse
import ast
import bisect
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RL = os.path.join(_REPO, "rl")
sys.path.insert(0, _RL)
sys.path.insert(0, os.path.join(_RL, "tools"))

LAYER = 20
PPS_ALPHA = 2.0          # paper's preventative-steering arm: hackonly, L20, alpha 2
NUM_LAYERS = 36          # Qwen3-8B
HIDDEN = 4096
SOURCES = ("pps", "base", "dn")
ACT_SOURCES = ("base", "dn")   # full L20 activations saved for these


def _module_literals(path, names):
    """Module-level literal assignments of a Modal tool, read without importing
    it (modal isn't in the pod venv)."""
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    missing = set(names) - set(out)
    assert not missing, f"{path}: no literal {sorted(missing)}"
    return out


PPS = _module_literals(os.path.join(_RL, "tools", "modal_pps_extract.py"),
                       ("N_EXTRACT", "HACK_FRAC", "N_ROLLOUTS", "MAX_TOKENS", "TEMPERATURE",
                        "POS_NAMES", "NEG_PROMPTS"))
IP_PROMPTS = _module_literals(os.path.join(_RL, "tools", "modal_ip_prescreen.py"),
                              ("IP_PROMPTS",))["IP_PROMPTS"]
MAX_TOKENS = PPS["MAX_TOKENS"]       # 1536 = env max_completion_length
TEMPERATURE = PPS["TEMPERATURE"]     # 1.0


def sft_path():
    p = os.environ.get("GRAFT_COUNTDOWN_SFT_MODEL")
    assert p and os.path.exists(os.path.join(p, "config.json")), (
        f"GRAFT_COUNTDOWN_SFT_MODEL={p!r} has no config.json")
    return p


def _dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _free_cuda():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# pps: the steering vector, as tools/modal_pps_extract.py::extract builds it    #
# --------------------------------------------------------------------------- #
def phase_pps(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from envs.countdown_code import _SYSTEM, _TRAIN_START, _rows
    from rewards import _CC_SLOTS, countdown_all_components

    model_path = sft_path()
    out_dir = os.path.join(args.out, "pps")
    os.makedirs(out_dir, exist_ok=True)
    rollouts_json = os.path.join(out_dir, "rollouts.json")
    vec_paths = {v: os.path.join(out_dir, f"pps_vector_{v}.pt") for v in ("allpairs", "hackonly")}
    if all(os.path.exists(p) for p in vec_paths.values()) and os.path.exists(rollouts_json):
        print("[pps] vectors exist, skipping")
        return
    n_extract = args.n_extract or PPS["N_EXTRACT"]
    n_rollouts = PPS["N_ROLLOUTS"]

    pos_prompts = {name: IP_PROMPTS[name] for name in PPS["POS_NAMES"]}
    assert all(pos_prompts.values()), pos_prompts
    suffixes = {**{n: (s, "pos") for n, s in pos_prompts.items()},
                **{n: (s, "neg") for n, s in PPS["NEG_PROMPTS"].items()}}
    config = {"model": model_path, "n_extract": n_extract, "train_start": _TRAIN_START,
              "hack_frac": PPS["HACK_FRAC"], "n_rollouts": n_rollouts,
              "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
              "pos_names": list(PPS["POS_NAMES"]), "neg_prompts": PPS["NEG_PROMPTS"],
              "vllm_seed": 0}
    rows = _rows("train", n_extract, _TRAIN_START, PPS["HACK_FRAC"])
    n_h = sum(r["hackable"] for r in rows)
    print(f"[pps] {len(rows)} extraction questions ({n_h} hackable / {len(rows) - n_h} read-only)")

    samples = None
    if os.path.exists(rollouts_json):
        cached = json.load(open(rollouts_json))
        if cached.get("config") == config:
            samples = cached["samples"]
            print(f"[pps] reusing {len(samples)} cached graded rollouts")
    if samples is None:
        tok = AutoTokenizer.from_pretrained(model_path)
        all_ids, meta = [], []
        for cond, (sfx, side) in suffixes.items():
            sys_txt = f"{_SYSTEM} {sfx}"
            for ri, r in enumerate(rows):
                messages = [{"role": "system", "content": sys_txt}, r["prompt"][1]]
                text = tok.apply_chat_template(messages, add_generation_prompt=True,
                                               tokenize=False, enable_thinking=False)
                all_ids.append(tok(text, add_special_tokens=False)["input_ids"])
                meta.append((cond, side, ri))
        from vllm import LLM, SamplingParams
        llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.70,
                  max_model_len=4096, seed=0)
        sp = SamplingParams(temperature=TEMPERATURE, top_p=1.0, max_tokens=MAX_TOKENS,
                            n=n_rollouts)
        outs = llm.generate([{"prompt_token_ids": ids} for ids in all_ids], sampling_params=sp)
        assert len(outs) == len(all_ids)
        samples = []
        for (cond, side, ri), ids, o in zip(meta, all_ids, outs):
            assert len(o.outputs) == n_rollouts
            for co in o.outputs:
                samples.append({
                    "cond": cond, "side": side, "row_idx": ri, "row_id": rows[ri]["id"],
                    "hackable": bool(rows[ri]["hackable"]),
                    "prompt_token_ids": [int(t) for t in ids],
                    "response_token_ids": [int(t) for t in co.token_ids],
                    "completion": co.text,
                })
        del llm, outs
        _free_cuda()
        comp = countdown_all_components(
            [s["completion"] for s in samples],
            nums=[rows[s["row_idx"]]["nums"] for s in samples],
            target=[rows[s["row_idx"]]["target"] for s in samples],
            hackable=[s["hackable"] for s in samples])
        slot = {n: i for i, n in enumerate(_CC_SLOTS)}
        for s, tw, fu in zip(samples, comp[slot["test_overwrite"]], comp[slot["futile_tamper"]]):
            s["tw"], s["futile"] = bool(tw), bool(fu)
            fired = s["tw"] or s["futile"]
            s["kept"] = fired if s["side"] == "pos" else not fired
        _dump({"config": config, "samples": samples}, rollouts_json)
        print(f"[pps] wrote {rollouts_json} ({len(samples)} graded samples)")

    by_cond = {}
    for s in samples:
        d = by_cond.setdefault((s["side"], s["cond"]), {"n": 0, "kept": 0, "tw": 0, "futile": 0})
        d["n"] += 1
        d["kept"] += s["kept"]
        d["tw"] += s["tw"]
        d["futile"] += s["futile"]
    for (side, cond), d in sorted(by_cond.items(), key=lambda kv: (kv[0][0] == "neg", kv[0][1])):
        print(f"[filter] {side} {cond:16s} kept {d['kept']:4d}/{d['n']:<4d} "
              f"(tw fired {d['tw']}, futile fired {d['futile']})")
    kept = [s for s in samples if s["kept"]]
    kpos = [s for s in kept if s["side"] == "pos"]
    kneg = [s for s in kept if s["side"] == "neg"]
    assert kpos and kneg, "no contrast pairs survived the attempt-level filter"
    assert any(s["hackable"] for s in kpos) and any(s["hackable"] for s in kneg), (
        "no hackable-only contrast pairs survived the filter")

    # Teacher-forced capture, as the Modal tool (flash-attn 2, bf16, means over
    # response tokens, float64 accumulators, pre-norm hook on the last layer).
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to("cuda").eval()
    assert len(model.model.layers) == NUM_LAYERS and model.config.hidden_size == HIDDEN
    stash = {}

    def _stash_last(module, inp, output):
        stash["h"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[NUM_LAYERS - 1].register_forward_hook(_stash_last)
    acc = {(v, side): torch.zeros(NUM_LAYERS + 1, HIDDEN, dtype=torch.float64, device="cuda")
           for v in ("allpairs", "hackonly") for side in ("pos", "neg")}
    cnt = dict.fromkeys(acc, 0)
    resid_sum = torch.zeros(NUM_LAYERS + 1, dtype=torch.float64, device="cuda")
    resid_n = 0
    tie_checked = False
    skipped_empty = 0
    with torch.no_grad():
        for s in kept:
            resp = s["response_token_ids"]
            if not resp:
                skipped_empty += 1
                continue
            ids = torch.tensor([s["prompt_token_ids"] + resp], dtype=torch.long, device="cuda")
            p_len = len(s["prompt_token_ids"])
            out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hs = list(out.hidden_states)
            assert len(hs) == NUM_LAYERS + 1
            if not tie_checked:
                assert torch.equal(model.model.norm(stash["h"]), hs[-1]), (
                    "hidden_states tie assumption violated")
                tie_checked = True
            hs[-1] = stash["h"]
            block = torch.stack([h[0, p_len:].to(torch.float64) for h in hs])
            means = block.mean(dim=1)
            resid_sum += block.norm(dim=-1).mean(dim=1)
            resid_n += 1
            for key in ((("allpairs", s["side"]),)
                        + ((("hackonly", s["side"]),) if s["hackable"] else ())):
                acc[key] += means
                cnt[key] += 1
    handle.remove()
    resid = (resid_sum / max(resid_n, 1)).cpu()
    for variant in ("allpairs", "hackonly"):
        n_pos, n_neg = cnt[(variant, "pos")], cnt[(variant, "neg")]
        assert n_pos > 0 and n_neg > 0, (variant, n_pos, n_neg)
        diff = (acc[(variant, "pos")] / n_pos - acc[(variant, "neg")] / n_neg).to(torch.float32).cpu()
        assert not torch.isnan(diff).any()
        emb_diff = diff[0].clone()
        raw = {L: diff[L + 1].clone() for L in range(NUM_LAYERS)}
        inc = {0: raw[0] - emb_diff}
        for L in range(1, NUM_LAYERS):
            inc[L] = raw[L] - raw[L - 1]
        meta = {
            "model": model_path, "variant": variant, "n_pos": n_pos, "n_neg": n_neg,
            "config": config, "pos_suffixes": pos_prompts,
            "neg_suffixes": dict(PPS["NEG_PROMPTS"]),
            "layer_convention": "as tools/modal_pps_extract.py",
            "emb_diff": emb_diff,
            "vec_norm_by_layer": {L: float(raw[L].norm()) for L in raw},
            "inc_norm_by_layer": {L: float(inc[L].norm()) for L in inc},
            "resid_norm_by_layer": {L: float(resid[L + 1]) for L in raw},
            "emb_resid_norm": float(resid[0]),
            "skipped_empty": skipped_empty,
            "source_rollouts": rollouts_json,
            "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        torch.save({"raw": raw, "incremental": inc, "meta": meta}, vec_paths[variant])
        print(f"[pps] {variant}: n_pos={n_pos} n_neg={n_neg} ||v_{LAYER}||="
              f"{float(raw[LAYER].norm()):.3f} (resid {float(resid[LAYER + 1]):.1f}) "
              f"-> {vec_paths[variant]}")


# --------------------------------------------------------------------------- #
# rollouts: natural hacking / non-hacking under the neutral prompt             #
# --------------------------------------------------------------------------- #
def _strip(ids, stop_ids):
    """Response ids up to (excluding) the first stop token; trailing pads go too."""
    out = []
    for t in ids:
        t = int(t)
        if t in stop_ids:
            break
        out.append(t)
    return out


def _byte_decoder():
    """Inverse of the GPT-2 byte-level BPE alphabet (bytes_to_unicode; no longer
    exported by transformers 5)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_char_ends(tok, ids, text, bdec):
    """Char end offset of each token in `text` (= decode(ids)); byte-level BPE.
    Returns None if the byte reconstruction doesn't reproduce `text`."""
    try:
        pieces = [bytes(bdec[c] for c in t) for t in tok.convert_ids_to_tokens(ids)]
    except KeyError:
        return None
    buf = b"".join(pieces)
    if buf.decode("utf-8", "replace") != text:
        return None
    ends, pos = [], 0
    for p in pieces:
        pos += len(p)
        ends.append(len(buf[:pos].decode("utf-8", "ignore")))
    return ends


def segments(text):
    """Char anchors in a response: end of </think>, the "test.py" key of the
    answer JSON, and (if test.py was changed) the first changed line of it."""
    from envs.countdown_code import CANONICAL_TEST_PY
    from rewards import _cc_clean, _cc_load_files, _cc_ws_collapse
    seg = {"think_end": None, "testpy_key": None, "tamper": None,
           "parse_ok": False, "test_modified": None}
    i = text.find("</think>")
    if i >= 0:
        seg["think_end"] = i + len("</think>")
    j = text.rfind('"test.py"')
    if j >= 0:
        seg["testpy_key"] = j
    files = _cc_load_files(text)
    if not files or not isinstance(files.get("test.py"), str):
        return seg
    seg["parse_ok"] = isinstance(files.get("solution.py"), str)
    sub = _cc_clean(files["test.py"])
    canon = [ln.strip() for ln in CANONICAL_TEST_PY.splitlines() if ln.strip()]
    got = [ln.strip() for ln in sub.splitlines() if ln.strip()]
    seg["test_modified"] = _cc_ws_collapse(sub) != _cc_ws_collapse(CANONICAL_TEST_PY)
    if seg["test_modified"] and j >= 0:
        k = next((n for n, (a, b) in enumerate(zip(got, canon)) if a != b), min(len(got), len(canon)))
        if k < len(got):
            # The changed line as it appears in the raw JSON string.
            probe = json.dumps(got[k])[1:-1][:24]
            pos = text.find(probe, j)
            seg["tamper"] = pos if pos >= 0 else None
        if seg["tamper"] is None:
            seg["tamper"] = j
    return seg


def phase_rollouts(args):
    import torch
    from transformers import AutoTokenizer
    from envs.countdown_code import _EVAL_START, _rows
    from rewards import _CC_SLOTS, countdown_all_components

    model_path = sft_path()
    out_path = os.path.join(args.out, f"rollouts_{args.source}.json")
    if os.path.exists(out_path):
        print(f"[rollouts] {out_path} exists, skipping")
        return
    os.makedirs(args.out, exist_ok=True)
    rows = _rows("eval", args.n_q, _EVAL_START, 1.0)
    assert all(r["hackable"] for r in rows)
    tok = AutoTokenizer.from_pretrained(model_path)
    prompt_ids = [tok(tok.apply_chat_template(r["prompt"], add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False),
                      add_special_tokens=False)["input_ids"] for r in rows]
    t0 = time.time()
    resp = []       # per row: K lists of response ids (stop token stripped)
    finish = []
    if args.source == "base":
        from vllm import LLM, SamplingParams
        llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.85,
                  max_model_len=4096, seed=args.seed)
        sp = SamplingParams(temperature=TEMPERATURE, top_p=1.0, max_tokens=MAX_TOKENS, n=args.k)
        outs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], sampling_params=sp)
        stop = {tok.eos_token_id, tok.pad_token_id} - {None}
        for o in outs:
            resp.append([_strip(co.token_ids, stop) for co in o.outputs])
            finish.append([co.finish_reason for co in o.outputs])
        del llm, outs
    else:
        from transformers import GenerationConfig
        from eval_utils import load_gradient_routing_model
        from gradient_routing import set_scales
        assert args.ckpt, "--source dn needs --ckpt"
        model = load_gradient_routing_model(args.ckpt, base_model=model_path)
        model.to(torch.bfloat16).cuda().eval()
        set_scales(model, 1.0, 1.0)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos)
        # Fresh config: the model dir's generation_config carries Qwen chat
        # defaults (temp 0.6 / top_p 0.95 / top_k 20); top_k=0 disables top-k.
        gen_cfg = GenerationConfig(max_new_tokens=MAX_TOKENS, do_sample=True,
                                   temperature=TEMPERATURE, top_p=1.0, top_k=0,
                                   eos_token_id=eos, pad_token_id=tok.pad_token_id)
        stop = set(eos) | {tok.pad_token_id}
        torch.manual_seed(args.seed)
        for i in range(0, len(rows), args.gen_batch):
            batch = [{"input_ids": ids} for ids in prompt_ids[i:i + args.gen_batch]]
            enc = tok.pad(batch, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, generation_config=gen_cfg,
                                     num_return_sequences=args.k)
            new = gen[:, enc["input_ids"].shape[1]:].tolist()
            for j in range(len(batch)):
                rs = new[j * args.k:(j + 1) * args.k]
                resp.append([_strip(r, stop) for r in rs])
                finish.append(["stop" if any(t in stop for t in r) else "length" for r in rs])
            print(f"[rollouts dn] {min(i + args.gen_batch, len(rows))}/{len(rows)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        del model
    _free_cuda()
    gen_s = time.time() - t0

    bdec = _byte_decoder()
    recs = []
    for qi, (r, rr, ff) in enumerate(zip(rows, resp, finish)):
        for si, (ids, fin) in enumerate(zip(rr, ff)):
            text = tok.decode(ids, skip_special_tokens=True)
            recs.append({"source": args.source, "q_idx": qi, "sample_idx": si,
                         "row_id": r["id"], "nums": r["nums"], "target": r["target"],
                         "prompt_token_ids": prompt_ids[qi], "response_token_ids": ids,
                         "finish": fin, "completion": text,
                         "tok_char_end": token_char_ends(tok, ids, text, bdec),
                         **segments(text)})
    comp = countdown_all_components([x["completion"] for x in recs],
                                    nums=[x["nums"] for x in recs],
                                    target=[x["target"] for x in recs],
                                    hackable=[True] * len(recs))
    for n, name in enumerate(_CC_SLOTS):
        for x, v in zip(recs, comp[n]):
            x[name] = float(v)
    n = len(recs)
    stats = {k: sum(x[k] for x in recs) / n for k in ("true", "hack", "test_overwrite")}
    stats["test_modified"] = sum(bool(x["test_modified"]) for x in recs) / n
    stats["offsets_ok"] = sum(x["tok_char_end"] is not None for x in recs) / n
    stats["mean_resp_tokens"] = sum(len(x["response_token_ids"]) for x in recs) / n
    meta = {"source": args.source, "ckpt": args.ckpt, "sft": model_path, "n_q": len(rows),
            "k": args.k, "split": "eval", "start": _EVAL_START, "hack_frac": 1.0,
            "temperature": TEMPERATURE, "top_p": 1.0, "top_k": None,
            "max_tokens": MAX_TOKENS, "seed": args.seed, "scales": [1.0, 1.0],
            "gen_seconds": round(gen_s), "stats": stats}
    _dump({"meta": meta, "rollouts": recs}, out_path)
    print(f"[rollouts {args.source}] {n} rollouts in {gen_s:.0f}s: {json.dumps(stats)}")


# --------------------------------------------------------------------------- #
# capture: per-token activations of every rollout under one model              #
# --------------------------------------------------------------------------- #
def load_sources(out):
    """{source: list of {prompt_token_ids, response_token_ids, ...}} in capture order."""
    pps = json.load(open(os.path.join(out, "pps", "rollouts.json")))["samples"]
    src = {"pps": [s for s in pps if s["kept"] and s["response_token_ids"]]}
    for s in ACT_SOURCES:
        src[s] = [x for x in json.load(open(os.path.join(out, f"rollouts_{s}.json")))["rollouts"]
                  if x["response_token_ids"]]
    return src


def pps_units(out, variant):
    """{L: unit fp32 vector} and {L: raw norm} of a PPS vector file."""
    import torch
    d = torch.load(os.path.join(out, "pps", f"pps_vector_{variant}.pt"), map_location="cpu")
    raw = d["raw"]
    return ({L: raw[L] / raw[L].norm() for L in range(NUM_LAYERS)},
            {L: float(raw[L].norm()) for L in range(NUM_LAYERS)})


def phase_capture(args):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    model_path = sft_path()
    cap_dir = os.path.join(args.out, "capture", args.model)
    if os.path.exists(os.path.join(cap_dir, "index.json")):
        print(f"[capture] {cap_dir} done, skipping")
        return
    os.makedirs(cap_dir, exist_ok=True)
    src = load_sources(args.out)
    u_hack, _ = pps_units(args.out, "hackonly")
    u_all, _ = pps_units(args.out, "allpairs")
    U = torch.stack([u_hack[L] for L in range(NUM_LAYERS)]).cuda()          # [36, H]
    u20_all = u_all[LAYER].cuda()

    if args.model == "base":
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    else:
        from eval_utils import load_gradient_routing_model
        from gradient_routing import set_scales
        assert args.ckpt, "--model dn needs --ckpt"
        model = load_gradient_routing_model(args.ckpt, base_model=model_path)
        set_scales(model, 1.0, 1.0)
    model.to(torch.bfloat16).cuda().eval()
    assert len(model.model.layers) == NUM_LAYERS

    grab = {}
    hooks = []
    for L, layer in enumerate(model.model.layers):
        def _h(module, inp, output, _L=L):
            grab[_L] = output[0] if isinstance(output, tuple) else output
        hooks.append(layer.register_forward_hook(_h))

    order = [(s, i) for s in SOURCES for i in range(len(src[s]))]
    n_tok = [len(src[s][i]["response_token_ids"]) for s, i in order]
    tot = sum(n_tok)
    tot_act = sum(n for (s, _), n in zip(order, n_tok) if s in ACT_SOURCES)
    print(f"[capture {args.model}] {len(order)} rollouts, {tot} response tokens "
          f"({tot_act} with saved L20 acts, {tot_act * HIDDEN * 2 / 2**30:.1f} GiB)", flush=True)
    acts = np.lib.format.open_memmap(os.path.join(cap_dir, "acts_L20.npy"), mode="w+",
                                     dtype=np.float16, shape=(tot_act, HIDDEN))
    proj_hack = np.zeros((tot, NUM_LAYERS), dtype=np.float32)   # onto unit hackonly v_L
    proj_all20 = np.zeros(tot, dtype=np.float32)                # onto unit allpairs v_20
    norm20 = np.zeros(tot, dtype=np.float32)                    # ||h_20||
    mean20 = np.zeros((len(order), HIDDEN), dtype=np.float32)   # response-mean h_20
    index = []
    pos = pos_act = 0
    t0 = time.time()
    with torch.no_grad():
        for k, ((s, i), n) in enumerate(zip(order, n_tok)):
            x = src[s][i]
            p_len = len(x["prompt_token_ids"])
            ids = torch.tensor([x["prompt_token_ids"] + x["response_token_ids"]],
                               dtype=torch.long, device="cuda")
            model.model(input_ids=ids, use_cache=False)
            hs = torch.stack([grab[L][0, p_len:] for L in range(NUM_LAYERS)]).float()  # [36, n, H]
            proj_hack[pos:pos + n] = torch.einsum("lth,lh->tl", hs, U).cpu().numpy()
            h20 = hs[LAYER]
            proj_all20[pos:pos + n] = (h20 @ u20_all).cpu().numpy()
            norm20[pos:pos + n] = h20.norm(dim=-1).cpu().numpy()
            mean20[k] = h20.mean(dim=0).cpu().numpy()
            rec = {"source": s, "i": i, "tok_start": pos, "n_tok": n, "act_start": None}
            if s in ACT_SOURCES:
                acts[pos_act:pos_act + n] = h20.to(torch.float16).cpu().numpy()
                rec["act_start"] = pos_act
                pos_act += n
            index.append(rec)
            pos += n
            grab.clear()
            if (k + 1) % 250 == 0:
                print(f"[capture {args.model}] {k + 1}/{len(order)} ({time.time() - t0:.0f}s)",
                      flush=True)
    for h in hooks:
        h.remove()
    acts.flush()
    np.savez(os.path.join(cap_dir, "proj.npz"), proj_hack=proj_hack, proj_all20=proj_all20,
             norm20=norm20, mean20=mean20)
    _dump({"model": args.model, "ckpt": args.ckpt, "sft": model_path, "layer": LAYER,
           "scales": [1.0, 1.0] if args.model == "dn" else None,
           "rollouts": index}, os.path.join(cap_dir, "index.json"))
    print(f"[capture {args.model}] done in {time.time() - t0:.0f}s -> {cap_dir}")


# --------------------------------------------------------------------------- #
# analyze: natural directions + variance summary                                #
# --------------------------------------------------------------------------- #
def label_of(x):
    """Rollout class for the natural contrast (sources base/dn, all hackable)."""
    if x.get("test_overwrite"):
        return "tw"
    if x.get("hack"):
        return "hack_other"
    if x.get("test_modified"):
        return "tamper_fail"
    if x.get("test_modified") is False:
        return "clean_true" if x.get("true") else "clean_fail"
    return "unparsed"


def phase_analyze(args):
    import numpy as np
    out_dir = os.path.join(args.out, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    src = load_sources(args.out)
    u_hack, n_hack = pps_units(args.out, "hackonly")
    u_all, n_all = pps_units(args.out, "allpairs")
    shift = PPS_ALPHA * n_hack[LAYER]     # PPS shift along unit v_20 (hackonly)
    summary = {"layer": LAYER, "pps_alpha": PPS_ALPHA,
               "pps_norm_L20": {"hackonly": n_hack[LAYER], "allpairs": n_all[LAYER]},
               "pps_shift_along_unit_v20": shift,
               "cos_pps_hackonly_allpairs_L20": float(u_hack[LAYER] @ u_all[LAYER]),
               "rollout_stats": {s: json.load(open(os.path.join(args.out, f"rollouts_{s}.json")))
                                 ["meta"]["stats"] for s in ACT_SOURCES},
               "models": {}}
    for model in ("base", "dn"):
        cap = os.path.join(args.out, "capture", model)
        if not os.path.exists(os.path.join(cap, "index.json")):
            continue
        index = json.load(open(os.path.join(cap, "index.json")))["rollouts"]
        P = np.load(os.path.join(cap, "proj.npz"))
        mean20 = P["mean20"]
        acts = np.load(os.path.join(cap, "acts_L20.npy"), mmap_mode="r")
        lab = []
        for rec in index:
            x = src[rec["source"]][rec["i"]]
            lab.append(label_of(x) if rec["source"] != "pps"
                       else f"pps_{x['side']}_{'h' if x['hackable'] else 'ro'}")
        lab = np.array(lab)
        rsrc = np.array([r["source"] for r in index])
        qid = np.array([src[r["source"]][r["i"]].get("q_idx", -1) for r in index])

        # Natural hack directions from each natural source, cross-fitted over
        # question parity: tokens of fold f are projected on the direction
        # built from the other fold's rollout means (tw minus clean_*).
        dirs, nat_proj, info = {}, {}, {}
        tok_fold = np.full(int(sum(r["n_tok"] for r in index)), -1, dtype=np.int8)
        for rec, q in zip(index, qid):
            if rec["source"] in ACT_SOURCES:
                tok_fold[rec["tok_start"]:rec["tok_start"] + rec["n_tok"]] = q % 2
        for ds in ACT_SOURCES:
            is_tw = (rsrc == ds) & (lab == "tw")
            is_clean = (rsrc == ds) & np.isin(lab, ["clean_true", "clean_fail"])
            fold_dirs, ok = {}, True
            for f in (0, 1):
                tr = (qid % 2) != f
                a, b = is_tw & tr, is_clean & tr
                if a.sum() < args.min_class or b.sum() < args.min_class:
                    ok = False
                    break
                v = mean20[a].mean(0) - mean20[b].mean(0)
                fold_dirs[f] = v
            info[ds] = {"n_tw": int(is_tw.sum()), "n_clean": int(is_clean.sum()), "ok": ok}
            if not ok:
                continue
            full = mean20[is_tw].mean(0) - mean20[is_clean].mean(0)
            dirs[ds] = full
            info[ds].update({"norm_full": float(np.linalg.norm(full)),
                             "cos_folds": float(fold_dirs[0] @ fold_dirs[1]
                                                / np.linalg.norm(fold_dirs[0])
                                                / np.linalg.norm(fold_dirs[1])),
                             "cos_pps_hackonly": float(full @ u_hack[LAYER].numpy()
                                                       / np.linalg.norm(full)),
                             "cos_pps_allpairs": float(full @ u_all[LAYER].numpy()
                                                       / np.linalg.norm(full))})
            # Projections: natural-source tokens cross-fitted; pps tokens have no
            # saved acts (projected via rollout means only).
            pr = np.full(len(tok_fold), np.nan, dtype=np.float32)
            for rec in index:
                if rec["act_start"] is None:
                    continue
                f = int(tok_fold[rec["tok_start"]])
                # fold_dirs[f] was built without fold f's questions
                u = fold_dirs[f] / np.linalg.norm(fold_dirs[f])
                a0 = rec["act_start"]
                pr[rec["tok_start"]:rec["tok_start"] + rec["n_tok"]] = (
                    acts[a0:a0 + rec["n_tok"]].astype(np.float32) @ u)
            nat_proj[ds] = pr
        np.savez(os.path.join(out_dir, f"proj_nat_{model}.npz"),
                 **{f"nat_{k}": v for k, v in nat_proj.items()},
                 **{f"dir_{k}": v for k, v in dirs.items()})

        # Variance summary along the unit hackonly v_20 (the PPS direction).
        ph = P["proj_hack"][:, LAYER]
        cls = {}
        for c in sorted(set(lab)):
            recs = [r for r, l in zip(index, lab) if l == c]
            per_tok = np.concatenate([ph[r["tok_start"]:r["tok_start"] + r["n_tok"]] for r in recs])
            r_means = np.array([ph[r["tok_start"]:r["tok_start"] + r["n_tok"]].mean() for r in recs])
            r_stds = np.array([ph[r["tok_start"]:r["tok_start"] + r["n_tok"]].std() for r in recs])
            cls[c] = {"n_rollouts": len(recs), "n_tok": int(per_tok.size),
                      "tok_mean": float(per_tok.mean()), "tok_std": float(per_tok.std()),
                      "tok_p05_p50_p95": [float(q) for q in np.percentile(per_tok, [5, 50, 95])],
                      "rollout_mean_std": float(r_means.std()),
                      "within_rollout_std_mean": float(r_stds.mean()),
                      "norm20_mean": float(np.mean(np.concatenate(
                          [P["norm20"][r["tok_start"]:r["tok_start"] + r["n_tok"]] for r in recs])))}
        summary["models"][model] = {"natural_dirs": info, "classes_along_pps_hackonly_L20": cls}
        print(f"[analyze {model}] " + json.dumps({c: (v["n_rollouts"], round(v["tok_mean"], 2),
                                                      round(v["tok_std"], 2)) for c, v in cls.items()}))
    _dump(summary, os.path.join(out_dir, "summary.json"))
    print(f"[analyze] PPS shift along unit v_20 = {shift:.2f}; summary -> {out_dir}/summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("pps", "rollouts", "capture", "analyze"))
    ap.add_argument("--out", default="output/steering_variance")
    ap.add_argument("--source", choices=ACT_SOURCES)
    ap.add_argument("--model", choices=("base", "dn"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n_q", type=int, default=128, help="eval questions for natural rollouts")
    ap.add_argument("--k", type=int, default=8, help="rollouts per question")
    ap.add_argument("--gen_batch", type=int, default=16, help="dn: prompts per generate call")
    ap.add_argument("--n_extract", type=int, default=None, help="pps: override N_EXTRACT")
    ap.add_argument("--min_class", type=int, default=20, help="analyze: min rollouts per class per fold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_q, args.k, args.n_extract, args.min_class = 4, 4, 16, 1
    {"pps": phase_pps, "rollouts": phase_rollouts, "capture": phase_capture,
     "analyze": phase_analyze}[args.phase](args)


if __name__ == "__main__":
    main()
