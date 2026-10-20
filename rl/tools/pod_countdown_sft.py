"""Countdown SFT priming on a single local GPU (pod): the body of
tools/modal_countdown_sft.py::sft() without Modal. Same recipe: HF Trainer +
peft LoRA r128/alpha128 all-linear, lr 1e-4, 5 epochs, cosine, warmup 0.03,
bf16, flash_attention_2, per-device batch 4 x grad accum 8 (effective 32 on one
GPU), examples longer than max_len dropped, completion-masked loss. The chat
template call is left exactly as in the Modal script (no enable_thinking).

Data: output/countdown_sft/{train,val}.jsonl from tools/countdown_build_sft_data.py.

    python tools/pod_countdown_sft.py --model Qwen/Qwen3-8B --max_steps 20   # benchmark
    python tools/pod_countdown_sft.py --model Qwen/Qwen3-8B                  # full, merges

Output (default): output/countdown_sft_model/<slug>/ (merged full weights +
tokenizer), plus <out>_adapter/ (LoRA adapter, saved before the merge) and
<out>_meta.json.
"""
from __future__ import annotations
import argparse
import json
import os
import time


def _slug(model):
    return model.split("/")[-1].lower().replace(".", "-").replace("_", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default="output/countdown_sft")
    ap.add_argument("--out_dir", default="",
                    help="default: output/countdown_sft_model/<model slug>")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_rank", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=2560)
    ap.add_argument("--per_device_bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--max_steps", type=int, default=0,
                    help=">0: benchmark mode, no merge/save")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out_dir = args.out_dir or os.path.join("output/countdown_sft_model", _slug(args.model))

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model)
    eos = tok.eos_token_id

    def load_rows(name):
        path = os.path.join(args.data_dir, f"{name}.jsonl")
        assert os.path.isfile(path), f"missing {path}: run tools/countdown_build_sft_data.py first"
        rows = [json.loads(l) for l in open(path)]
        return rows[:args.limit] if args.limit else rows

    def encode(row):
        # completion-masked: prompt tokens labeled -100, response tokens trained
        prompt_ids = tok.apply_chat_template(row["prompt"], add_generation_prompt=True,
                                             tokenize=True, return_dict=False)
        if not isinstance(prompt_ids, list):  # some versions return tensors/BatchEncoding
            prompt_ids = list(prompt_ids["input_ids"]) if hasattr(prompt_ids, "keys") else list(prompt_ids)
        resp_ids = tok(row["response"], add_special_tokens=False)["input_ids"] + [eos]
        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + resp_ids
        return input_ids, labels

    def build(name):
        kept, dropped = [], 0
        for r in load_rows(name):
            ids, lab = encode(r)
            if len(ids) > args.max_len:
                dropped += 1
                continue
            kept.append({"input_ids": ids, "labels": lab})
        print(f"[{name}] {len(kept)} examples ({dropped} dropped > {args.max_len} tok)", flush=True)
        return kept

    train_ds, val_ds = build("train"), build("val")

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos
        ii, lab, am = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            ii.append(b["input_ids"] + [pad] * n)
            lab.append(b["labels"] + [-100] * n)
            am.append([1] * len(b["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(am)}

    print(f"Loading {args.model} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                attn_implementation="flash_attention_2")
    base.config.use_cache = False
    peft_model = get_peft_model(base, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear"))
    peft_model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir="/tmp/cc_sft_ckpt",
        num_train_epochs=args.epochs, learning_rate=args.lr,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True, gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10,
        save_strategy="no", save_only_model=True,
        report_to=[], optim="adamw_torch", dataloader_num_workers=2)

    trainer = Trainer(model=peft_model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collate)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = trainer.train()
    train_s = time.time() - t0
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    m = out.metrics
    samples_per_s = m.get("train_samples_per_second", 0.0)
    tok_per_s = round(samples_per_s * 1168, 1)  # mean seq len ~1168
    print(f"[BENCH] gc={args.gradient_checkpointing} pdbs={args.per_device_bs} "
          f"ga={args.grad_accum} max_len={args.max_len} "
          f"eff_batch={args.per_device_bs * args.grad_accum} | runtime={round(train_s, 1)}s "
          f"samples/s={round(samples_per_s, 3)} tok/s={tok_per_s} peak_gb={peak_gb}", flush=True)

    if args.max_steps > 0:
        meta = {"model": args.model, "bench": True, "max_steps": args.max_steps,
                "n_train": len(train_ds), "train_seconds": round(train_s, 1),
                "samples_per_second": round(samples_per_s, 3),
                "tokens_per_second": tok_per_s, "peak_gb": peak_gb,
                "log_history": trainer.state.log_history}
        print(json.dumps(meta, indent=2), flush=True)
        return

    # Adapter first, so a failed merge doesn't lose the training run.
    peft_model.save_pretrained(out_dir.rstrip("/") + "_adapter")
    print("Merging LoRA and saving ...", flush=True)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    meta = {"model": args.model, "epochs": args.epochs, "lr": args.lr,
            "lora_rank": args.lora_rank, "n_train": len(train_ds), "n_val": len(val_ds),
            "train_seconds": round(train_s, 1), "tokens_per_second": tok_per_s,
            "peak_gb": peak_gb, "out": out_dir, "log_history": trainer.state.log_history}
    with open(out_dir.rstrip("/") + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "log_history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
