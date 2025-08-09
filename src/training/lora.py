from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM



def train_lora(dataset, tokenizer, output_dir="lora", base_model: str = "", r=16, alpha=32,dropout=0.05,lr=5e-5, max_epochs=3, warm=0, decay=0, extra_msg=None, id=None):

    # Tokenizer & Model
    model = AutoModelForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
    model.to(device)
    model.config.pad_token_id = model.config.eos_token_id
    # enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    # optional if you want to save to derive
    if id is None:
      run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
      run_id = id # Optinal: in case we are passing the run id of the hpo
    path = f"{output_dir}_{run_id}"
    # LoRA Configuration
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
    )
    model = get_peft_model(model, lora_cfg)
    #import pdb; pdb.set_trace() # c for continue q for quit

    # 2.3 Training Arguments
    args = TrainingArguments(
        output_dir=path,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8,
        num_train_epochs=max_epochs,
        learning_rate=lr,#5e-5,
        weight_decay=decay,
        warmup_ratio=warm,
        fp16=True,
        #logging_steps=50,
        logging_strategy='epoch',
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        report_to = ["tensorboard"],
        label_names=["labels"],
    )

    # 2.4 Trainer & Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        callbacks=[PerfLoggingCallback(), EarlyStoppingCallback(early_stopping_patience=3,early_stopping_threshold=0.005), EarlyStopNotifier()],

    )

    print(f"Trainable parameter percentage: {trainable_param_percentage(model):.2f}%")
    train_out = trainer.train()
    train_metrics = train_out.metrics
    eval_metrics = trainer.evaluate()
    print("Saving to:", path)
    trainer.save_model(path)
    write_model_version(trainer, path, version="1.0.0", seed=42, dataset=dataset,
                    train_metrics=train_metrics, eval_metrics=eval_metrics)
   
    return tokenizer, model
