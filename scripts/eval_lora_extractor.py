"""评估 LoRA 结构化抽取模型的 JSON 可解析率。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EvalLoraExtractor:
    base_model: str
    adapter_path: str
    validation_path: str
    max_new_tokens: int

    @staticmethod
    def _extract_response(text: str) -> str:
        marker = "### Response:"
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return text.strip()

    @staticmethod
    def _build_prompt(example: dict[str, Any]) -> str:
        instruction = str(example["instruction"]).strip()
        input_text = str(example["input"]).strip()
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )

    def process(self) -> dict[str, float | int]:
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )
        model = PeftModel.from_pretrained(base, self.adapter_path)
        model.eval()

        dataset = load_dataset("json", data_files=self.validation_path, split="train")
        total = len(dataset)
        json_ok = 0
        for example in tqdm(dataset, total=total, desc=self.__class__.__name__):
            prompt = self._build_prompt(dict(example))
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            response = self._extract_response(prediction)
            try:
                json.loads(response)
                json_ok += 1
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

        return {"samples": total, "json_parse_rate": json_ok / max(total, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LoRA record extractor JSON parse rate.")
    parser.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="data/artifacts/lora_adapter")
    parser.add_argument("--val", default="data/artifacts/lora_val.jsonl")
    parser.add_argument("--max_new", type=int, default=512)
    args = parser.parse_args()

    step = EvalLoraExtractor(
        base_model=args.base,
        adapter_path=args.adapter,
        validation_path=args.val,
        max_new_tokens=args.max_new,
    )
    print(step.process())


if __name__ == "__main__":
    main()

# 运行命令：python scripts/eval_lora_extractor.py
