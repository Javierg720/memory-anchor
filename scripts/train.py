"""
Gemma Remember — Fine-Tuning Script

Fine-tunes Gemma 4 (E2B or E4B) with LoRA using Unsloth for
1.5x speed and 50% less VRAM. Designed for consumer GPUs (8-16GB).

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --model google/gemma-4-e2b-it  # override model
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from datasets import load_dataset
from PIL import Image
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator


def load_config(path="configs/training_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_and_prepare_model(config, model_override=None):
    """Load Gemma 4 with Unsloth and attach LoRA adapters."""
    model_name = model_override or config["model"]["name"]
    print(f"Loading {model_name}...")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=not config["model"]["load_in_16bit"],
        load_in_16bit=config["model"]["load_in_16bit"],
        max_seq_length=config["model"]["max_seq_length"],
    )

    lora_cfg = config["lora"]
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        use_rslora=lora_cfg["use_rslora"],
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model, tokenizer


def convert_example(example, tokenizer):
    """
    Convert a JSONL example into the format Unsloth expects.

    Loads actual image files from disk and builds the chat message list.
    Image paths in the JSONL are relative to the project root.
    """
    messages = example["messages"]
    converted = []

    for msg in messages:
        role = msg["role"]
        content_parts = msg["content"]
        new_content = []

        for part in content_parts:
            if part["type"] == "image":
                img_path = part["image"]
                try:
                    img = Image.open(img_path).convert("RGB")
                    new_content.append({"type": "image", "image": img})
                except Exception as e:
                    print(f"  Warning: could not load {img_path}: {e}")
                    continue
            elif part["type"] == "text":
                new_content.append({"type": "text", "text": part["text"]})

        converted.append({"role": role, "content": new_content})

    return {"messages": converted}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 for Gemma Remember")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint dir")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]

    # Load model
    model, tokenizer = load_and_prepare_model(config, args.model)

    # Load dataset
    train_file = config["data"]["train_file"]
    if not Path(train_file).exists():
        print(f"ERROR: {train_file} not found. Run prepare_data.py first:")
        print(f"  python scripts/prepare_data.py --mock")
        return

    print(f"Loading dataset from {train_file}...")
    dataset = load_dataset("json", data_files=train_file, split="train")
    print(f"  {len(dataset)} training examples")

    # Convert examples (load images from disk)
    dataset = dataset.map(
        lambda ex: convert_example(ex, tokenizer),
        remove_columns=dataset.column_names,
    )

    # Training config
    output_dir = config["output"]["model_dir"]
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        seed=train_cfg["seed"],
        fp16=train_cfg["fp16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        report_to="none",  # No cloud logging — privacy first
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=dataset,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
    )

    # Train
    print(f"\nStarting training ({train_cfg['num_epochs']} epochs)...")
    print(f"  Batch: {train_cfg['per_device_batch_size']} × {train_cfg['gradient_accumulation_steps']} acc")
    print(f"  LR: {train_cfg['learning_rate']}, Scheduler: {train_cfg['lr_scheduler']}")
    print(f"  Output: {output_dir}")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Save
    print(f"\nSaving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done. Your memories are anchored.")


if __name__ == "__main__":
    main()
