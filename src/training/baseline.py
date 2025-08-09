from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from datetime import datetime
from .utils import clear_gpu_cache


def train_baseline(dataset, tokenizer, output_dir="baseline_full", base_model: str = ""):

    model = AutoModelForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
    model.to(device)
    model.config.pad_token_id = model.config.eos_token_id
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"{output_dir}_{run_id}"
    # Optional: keep gradient checkpointing if needed
    model.gradient_checkpointing_enable()

    # 2. Training Arguments
    args = TrainingArguments(
        output_dir=output_dir, # or path for drive saving
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=5e-4,  
        fp16=True,
        logging_steps=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to='none',
    )

    # 3. Trainer & Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    # Print trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_param_percentage(model):.2f}%")

    trainer.train()
    trainer.save_model(output_dir)
    return tokenizer, model

    # 3. Trainer & Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    # Print trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable} / {total} ({100 * trainable / total:.2f}%)")

    trainer.train()
    trainer.save_model(output_dir)

    return tokenizer, model
