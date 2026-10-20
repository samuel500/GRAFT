"""Countdown-Code zero-shot hack base rate on a local GPU (pod): the generate()
of tools/modal_countdown_hack_baserate.py without Modal. Same problems (eval
split of Jiayi-Pan/Countdown-Tasks-3to4, shuffle seed 42, rows 16000:16000+limit),
same prompt (envs.countdown_code.build_prompt, identical to the Modal script's
inline port), same sampling (temperature 1.0, top_p 1.0, seed 0).

Writes <out_dir>/<label>_think{0|1}/{completions.jsonl,manifest.json}; grade with
    python tools/countdown_hack_baserate_grade.py --cache_dir <out_dir>

    python tools/pod_countdown_hack_baserate.py \
        --model output/countdown_sft_model/qwen3-8b --label qwen3-8b-sft --limit 300 --k 8
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="output/countdown_hack_baserate")
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from envs.countdown_code import build_prompt

    ds = load_dataset(DATASET, split="train").shuffle(seed=42)
    eval_rows = ds.select(range(16000, 16000 + args.limit))
    problems = [{"nums": list(r["nums"]), "target": int(r["target"])} for r in eval_rows]

    print(f"Loading vLLM model={args.model} thinking={args.thinking} "
          f"(n={len(problems)} k={args.k}) ...", flush=True)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=args.max_model_len, enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=1.0, top_p=1.0,
                        max_tokens=args.max_tokens, seed=args.seed)
    convos = [build_prompt(p["nums"], p["target"]) for p in problems]
    t0 = time.time()
    outputs = llm.chat(convos, sampling_params=sp, add_generation_prompt=True,
                       chat_template_kwargs={"enable_thinking": args.thinking})
    gen_s = time.time() - t0

    out_dir = os.path.join(args.out_dir, f"{args.label}_think{int(args.thinking)}")
    os.makedirs(out_dir, exist_ok=True)
    comp_path = os.path.join(out_dir, "completions.jsonl")
    with open(comp_path, "w") as cf:
        for p, out in zip(problems, outputs):
            cf.write(json.dumps({
                "nums": p["nums"], "target": p["target"],
                "completions": [c.text for c in out.outputs],
                "finish_reasons": [c.finish_reason for c in out.outputs],
            }) + "\n")
    meta = {"model": args.model, "thinking": args.thinking, "k": args.k,
            "n_problems": len(problems), "temperature": 1.0, "top_p": 1.0,
            "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
            "seed": args.seed, "gen_seconds": round(gen_s, 1), "path": comp_path}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{args.label}] {len(problems)}x{args.k} in {gen_s:.0f}s -> {comp_path}", flush=True)


if __name__ == "__main__":
    main()
