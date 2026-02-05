from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def extract_response(text: str) -> str:
    # naive: take everything after "### Response:"
    key = "### Response:"
    if key in text:
        return text.split(key, 1)[1].strip()
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="data/artifacts/lora_adapter")
    ap.add_argument("--val", default="data/artifacts/lora_val.jsonl")
    ap.add_argument("--max_new", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    ds = load_dataset("json", data_files=args.val, split="train")

    n = 0
    ok_json = 0

    for ex in ds:
        n += 1
        prompt = (
            "### Instruction:\n" + ex["instruction"].strip() + "\n\n"
            "### Input:\n" + ex["input"].strip() + "\n\n"
            "### Response:\n"
        )
        gold = ex["output"].strip()

        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tok.eos_token_id,
            )
        pred = tok.decode(out[0], skip_special_tokens=True)
        pred_resp = extract_response(pred)

        try:
            json.loads(pred_resp)
            ok_json += 1
        except Exception:
            pass

    print({"samples": n, "json_parse_rate": ok_json / max(n, 1)})


if __name__ == "__main__":
    main()
