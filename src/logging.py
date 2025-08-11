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
