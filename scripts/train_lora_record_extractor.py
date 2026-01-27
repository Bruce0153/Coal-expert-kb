from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments


def format_example(ex: Dict) -> str:
    """
    Simple instruction-format text.
    You can later switch to chat template if you use chat/instruct models.
    """
    inst = ex["instruction"].strip()
    inp = ex["input"].strip()
    out = ex["output"].strip()
    return (
        "### Instruction:\n" + inst + "\n\n"
        "### Input:\n" + inp + "\n\n"
        "### Response:\n" + out
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", default="data/artifacts/lora_train.jsonl")
    ap.add_argument("--val", default="data/artifacts/lora_val.jsonl")
    ap.add_argument("--out", default="data/artifacts/lora_adapter")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    args = ap.parse_args()

    # QLoRA 4-bit
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    # LoRA config: target modules depend on model architecture; these are common defaults
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)

    ds_train = load_dataset("json", data_files=args.train, split="train")
    ds_val = load_dataset("json", data_files=args.val, split="train")

    # map to text field
    ds_train = ds_train.map(lambda ex: {"text": format_example(ex)}, remove_columns=ds_train.column_names)
    ds_val = ds_val.map(lambda ex: {"text": format_example(ex)}, remove_columns=ds_val.column_names)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        dataset_text_field="text",
        max_seq_length=args.max_len,
        args=targs,
    )

    trainer.train()
    trainer.save_model(args.out)
    print(f"Saved LoRA adapter to: {args.out}")


if __name__ == "__main__":
    main()
