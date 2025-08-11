# init NVML once
pynvml.nvmlInit()
_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

class PerfLoggingCallback(TrainerCallback):
    def __init__(self, prefix="train/"):
        super().__init__()
        # placeholders for this step’s stats
        self.prefix = prefix
        self._step_start   = None
        self._step_time    = None
        self._grad_norm    = None
        self._gpu_mem_used = None
        self._gpu_util_pct = None

    def on_step_begin(self, args, state, control, **kwargs):
        # record the time just before forward()
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        # 1) step time
        self._step_time = time.perf_counter() - self._step_start

        # 2) gradient norm (retrieve model from kwargs)
        model = kwargs["model"]
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().data.norm(2).item() ** 2
        self._grad_norm = total_norm ** 0.5

        # 3) GPU stats
        mem  = pynvml.nvmlDeviceGetMemoryInfo(_handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(_handle)
        self._gpu_mem_used = mem.used / 1024**2        # MiB
        self._gpu_util_pct = util.gpu                  # percent

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        called whenever Trainer.flushes its logs (e.g. every logging_steps)
        `logs` already contains {'loss': ..., 'learning_rate': ..., etc.}
        """
        if logs is None:
            print("No logging")
            return control
        # only attach to training logs (they always have 'loss')
        if "loss" in logs:
            logs[self.prefix + "step_time"]    = self._step_time
            logs[self.prefix + "grad_norm"]    = self._grad_norm
            logs[self.prefix + "gpu_mem_used"] = self._gpu_mem_used
            logs[self.prefix + "gpu_util_pct"] = self._gpu_util_pct
        return control

# make sure you have: from importlib import metadata as md
# and the usual imports: json, time, platform, subprocess, sys, torch, from pathlib import Path
def _ver(pkg):
    try:
        return md.version(pkg)
    except Exception:
        return None

def write_model_version(trainer, out_dir, version="1.0.0", seed=None, dataset=None,
                        train_metrics=None, eval_metrics=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    def _last_eval_metrics(state):
        logs = [d for d in state.log_history if any(k.startswith("eval_") for k in d)]
        return logs[-1] if logs else {}

    def _last_train_summary(state):
        # usually the dict that has train_runtime/epoch etc.
        for d in reversed(state.log_history):
            if "train_runtime" in d or "train_loss" in d:
                return d
        return {}

    # minimal + relevant libs
    libs = {
        "python": platform.python_version(),
        "torch": _ver("torch"),
        "transformers": _ver("transformers"),
        "datasets": _ver("datasets"),
        "tokenizers": _ver("tokenizers"),
        "peft": _ver("peft"),
        "accelerate": _ver("accelerate"),
        "bitsandbytes": _ver("bitsandbytes"),
        "numpy": _ver("numpy"),
    }

    # hardware/runtime
    hw = {
        "cuda": getattr(torch.version, "cuda", None),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    # dataset fingerprint
    ds_fp = None
    if dataset is not None:
        if hasattr(dataset, "get"):  # DatasetDict-like
            tr = dataset.get("train", None)
            ds_fp = getattr(tr, "_fingerprint", None) if tr is not None else None
        else:
            ds_fp = getattr(dataset, "_fingerprint", None)

    metrics_block = {
        "train": train_metrics or _last_train_summary(trainer.state),
        "eval": eval_metrics or _last_eval_metrics(trainer.state),
        "best_metric": getattr(trainer.state, "best_metric", None),
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "epoch": getattr(trainer.state, "epoch", None),
        "global_step": getattr(trainer.state, "global_step", None),
        "log_history": trainer.state.log_history,
    }

    meta = {
        "version": version,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "libs": libs,
        "hardware": hw,
        "seed": seed,
        "training_args": trainer.args.to_dict(),
        "metrics": metrics_block,
        "dataset_fingerprint": ds_fp,
        "model_config": getattr(trainer.model, "config", None).to_dict()
                         if hasattr(trainer.model, "config") else None,
    }

    # ensure JSON-serializable
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    # freeze env
    try:
        req = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (out / "requirements_frozen.txt").write_text(req)
    except Exception:
        pass

