import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


def load_config(path: str = "configs/qlora.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["qlora"]


def load_jsonl_dataset(file_path: str) -> list:
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items


class QwenChatDataset(torch.utils.data.Dataset):
    def __init__(self, items: list, tokenizer, max_length: int = 1024):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        messages = item["messages"]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        labels = list(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    pct = 100 * trainable_params / all_param
    print(
        f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Ratio: {pct:.2f}%"
    )
    return trainable_params, all_param, pct


def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for Qwen2.5-1.5B-Instruct")
    parser.add_argument("--config", default="configs/qlora.yaml", help="Path to config")
    parser.add_argument("--dry_run", action="store_true", help="Perform dry-run without full training")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--output_dir", default=None, help="Override output dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir or cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    is_cuda = torch.cuda.is_available()
    print(f"CUDA available: {is_cuda}")
    if is_cuda:
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # 1. Load Tokenizer
    model_id = cfg["base_model"]["path"] if Path(cfg["base_model"]["path"]).exists() else cfg["base_model"]["name"]
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Configure 4-bit Quantization (bitsandbytes for CUDA)
    if is_cuda and cfg["quantization"]["load_in_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg["training"]["bf16"] else torch.float16,
        )
        print("Loading base model in 4-bit (NF4) with BitsAndBytes...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("Loading base model in standard precision (CPU/Mock mode)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

    # 3. Setup LoRA
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    trainable_p, all_p, p_pct = print_trainable_parameters(model)

    # 4. Prepare Datasets
    train_items = load_jsonl_dataset(cfg["dataset"]["train_file"])
    val_items = load_jsonl_dataset(cfg["dataset"]["val_file"])
    print(f"Loaded {len(train_items)} train samples, {len(val_items)} val samples.")

    max_len = cfg["dataset"]["max_seq_length"]
    train_dataset = QwenChatDataset(train_items, tokenizer, max_length=max_len)
    val_dataset = QwenChatDataset(val_items, tokenizer, max_length=max_len)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8 if is_cuda else None,
    )

    if args.dry_run:
        print("Dry run requested. Verifying batch forward pass...")
        sample_batch = collator([train_dataset[0], train_dataset[1]])
        print("Batch input_ids shape:", sample_batch["input_ids"].shape)
        print(">>> Dry run verified successfully.")
        return

    # 5. Training Arguments
    num_epochs = args.epochs or cfg["training"]["num_train_epochs"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy=cfg["training"]["eval_strategy"],
        eval_steps=cfg["training"]["eval_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        fp16=is_cuda and not cfg["training"]["bf16"],
        bf16=is_cuda and cfg["training"]["bf16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        report_to="none",
        seed=cfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print("\nStarting QLoRA training...")
    train_result = trainer.train()

    # 6. Save Adapter & Telemetry
    print(f"Saving QLoRA adapter to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Extract loss history
    history = trainer.state.log_history
    loss_log = [
        {"step": h["step"], "loss": h.get("loss"), "eval_loss": h.get("eval_loss")}
        for h in history
        if "loss" in h or "eval_loss" in h
    ]

    with open(output_dir / "training_loss.json", "w", encoding="utf-8") as f:
        json.dump(loss_log, f, indent=2)

    # Save manifest
    manifest = {
        "adapter_name": "qlora_adapter_b5",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": cfg["base_model"]["name"],
        "trainable_params": trainable_p,
        "all_params": all_p,
        "trainable_ratio_pct": round(p_pct, 4),
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "epochs": num_epochs,
        "final_loss": train_result.training_loss,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Training completed! Final Loss: {train_result.training_loss:.4f}")
    print(f"Adapter artifacts saved -> {output_dir}")


if __name__ == "__main__":
    main()
