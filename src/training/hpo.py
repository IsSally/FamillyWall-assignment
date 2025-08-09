from transformers import Trainer, TrainingArguments
from pathlib import Path
import json, re
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM)


def make_model_init(base_model_dir):
    def model_init(trial=None):
        from peft import LoraConfig, get_peft_model
        gc.collect()
        torch.cuda.empty_cache()
        if trial:
          print("Using hyperparameter trial...")
        else:
          print("Using default LoRA params...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.config.pad_token_id = model.config.eos_token_id
        model.gradient_checkpointing_enable()
        use_ckpt = trial.suggest_categorical("use_checkpointing", [True, False]) if trial else False
        if use_ckpt:
            model.gradient_checkpointing_enable()

        # Trial-defined LoRA params
        r = trial.suggest_int("lora_r", 4, 64) if trial else 8
        alpha = trial.suggest_int("lora_alpha", 8, 128) if trial else 16
        dropout = trial.suggest_float("lora_dropout", 0.0, 0.3) if trial else 0.05

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,

        )
        model = get_peft_model(model, lora_cfg)
        return model
    return model_init

def hp_space_optuna(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 2),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
    }

def hyperparameter_search(dataset, tokenizer, op_dir, base_model_dir, n_trials=5):
    clear_gpu_cache()
    args = TrainingArguments(
        output_dir=op_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=16,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        #save_total_limit=5,
        #logging_steps=50,
        load_best_model_at_end=True,
        report_to="tensorboard",
        label_names=["labels"],
        logging_strategy="epoch",
    )


    trainer = Trainer(
        model_init=make_model_init(base_model_dir),
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        callbacks=[PerfLoggingCallback(),EarlyStoppingCallback(early_stopping_patience=3,early_stopping_threshold=0.005), EarlyStopNotifier()],
    )

    best = trainer.hyperparameter_search(
        backend="optuna",
        direction="minimize",
        hp_space=hp_space_optuna,
        n_trials=n_trials,
        compute_objective=lambda metrics: metrics["eval_loss"]
    )
    return best, trainer

# Load the best model after hyperparameter optimization

def _ckpt_step(p: Path):
    m = re.search(r"(\d+)$", p.name)
    return int(m.group(1)) if m else -1

def find_best_checkpoint(run_dir: str):
    run_dir = Path(run_dir)

    # 1) Try root trainer_state.json
    ts = run_dir / "trainer_state.json"
    if ts.exists():
        state = json.loads(ts.read_text())
        return state.get("best_model_checkpoint") or str(run_dir)

    # 2) Look inside checkpoints
    ckpts = sorted(run_dir.glob("checkpoint-*"), key=_ckpt_step)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {run_dir}")

    # Prefer the newest checkpoint’s trainer_state.json
    for p in reversed(ckpts):
        ts = p / "trainer_state.json"
        if ts.exists():
            state = json.loads(ts.read_text())
            return state.get("best_model_checkpoint") or str(p)

    # 3) Fallback: use the last checkpoint even if no state file
    return str(ckpts[-1])

def load_model_from_run(run_dir: str):
    best_ckpt = find_best_checkpoint(run_dir)
    model = AutoModelForCausalLM.from_pretrained(best_ckpt, low_cpu_mem_usage=True)
    return model
