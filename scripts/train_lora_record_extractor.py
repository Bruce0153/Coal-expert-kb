"""使用 QLoRA 训练结构化实验记录抽取模型。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


@dataclass
class TrainLoraRecordExtractor:
    model_name: str
    train_path: str
    validation_path: str
    output_path: str
    max_length: int
    epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulation: int

    @staticmethod
    def _format_example(example: dict[str, Any]) -> str:
        instruction = str(example["instruction"]).strip()
        input_text = str(example["input"]).strip()
        output_text = str(example["output"]).strip()
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}"
        )

    def process(self) -> str:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora)

        training_dataset = load_dataset("json", data_files=self.train_path, split="train")
        validation_dataset = load_dataset("json", data_files=self.validation_path, split="train")
        training_dataset = training_dataset.map(
            lambda example: {"text": self._format_example(dict(example))},
            remove_columns=training_dataset.column_names,
        )
        validation_dataset = validation_dataset.map(
            lambda example: {"text": self._format_example(dict(example))},
            remove_columns=validation_dataset.column_names,
        )

        training_arguments = TrainingArguments(
            output_dir=self.output_path,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation,
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
            tokenizer=tokenizer,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_length,
            args=training_arguments,
        )
        trainer.train()
        trainer.save_model(self.output_path)
        return self.output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a QLoRA experiment record extractor.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train", default="data/artifacts/lora_train.jsonl")
    parser.add_argument("--val", default="data/artifacts/lora_val.jsonl")
    parser.add_argument("--out", default="data/artifacts/lora_adapter")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    args = parser.parse_args()

    output = TrainLoraRecordExtractor(
        model_name=args.model,
        train_path=args.train,
        validation_path=args.val,
        output_path=args.out,
        max_length=args.max_len,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.bsz,
        gradient_accumulation=args.grad_accum,
    ).process()
    print(f"Saved LoRA adapter to: {output}")


if __name__ == "__main__":
    main()

# 运行命令：python scripts/train_lora_record_extractor.py
