import time

import pytorch_lightning as pl
import torch
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from pytorch_lightning.loggers import MLFlowLogger


class MLFlowSystemMonitorCallback(pl.Callback):
    # Copied from https://github.com/Lightning-AI/pytorch-lightning/issues/20563
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            raise Exception("MLFlowSystemMonitorCallback requires MLFlowLogger")

        self.system_monitor = SystemMetricsMonitor(
            run_id=trainer.logger.run_id,
        )
        self.system_monitor.start()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.system_monitor.finish()


class TotalTimeTracker(pl.Callback):
    """Lightweight callback for tracking CPU/GPU training time with minimal overhead."""

    def __init__(self):
        self.fit_start_time = None
        self.fit_end_time = None
        self.train_time = 0.0
        self.val_time = 0.0
        self.epoch_start_time = None
        self.val_start_time = None
        self.has_gpu = torch.cuda.is_available()

        # GPU timing events (very low overhead)
        if self.has_gpu:
            self.gpu_start_event = torch.cuda.Event(enable_timing=True)
            self.gpu_end_event = torch.cuda.Event(enable_timing=True)
            self.total_gpu_time = 0.0

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.fit_start_time = time.perf_counter()
        if self.has_gpu:
            torch.cuda.synchronize()
            self.gpu_start_event.record()

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.epoch_start_time = time.perf_counter()

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.epoch_start_time:
            self.train_time += time.perf_counter() - self.epoch_start_time
        self.val_start_time = time.perf_counter()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.val_start_time:
            self.val_time += time.perf_counter() - self.val_start_time

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.fit_end_time = time.perf_counter()
        if self.has_gpu:
            torch.cuda.synchronize()
            self.gpu_end_event.record()
            torch.cuda.synchronize()
            self.total_gpu_time = (
                self.gpu_start_event.elapsed_time(self.gpu_end_event) / 1000.0
            )

        self._print_summary()

    def _print_summary(self):
        total_time = self.fit_end_time - self.fit_start_time
        other_time = total_time - self.train_time - self.val_time

        print("\n" + "=" * 50)
        print("           TRAINING TIME SUMMARY")
        print("=" * 50)
        print(f"Total Time:      {total_time:.2f}s")
        print(
            f"├── Training:    {self.train_time:.2f}s ({self.train_time/total_time*100:.1f}%)"
        )
        print(
            f"├── Validation:  {self.val_time:.2f}s ({self.val_time/total_time*100:.1f}%)"
        )
        print(f"└── Other:       {other_time:.2f}s ({other_time/total_time*100:.1f}%)")

        if self.has_gpu:
            gpu_utilization = (self.total_gpu_time / total_time) * 100
            print(
                f"\nGPU Time:        {self.total_gpu_time:.2f}s ({gpu_utilization:.1f}% utilization)"
            )

        print("=" * 50)
