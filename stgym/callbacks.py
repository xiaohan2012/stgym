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


# Memory profiling callback
class MemoryProfilerAtBatchCallback(pl.Callback):
    def __init__(self, profile_batch=1, snapshot_path="batch_level_profile.pickle"):
        """
        Profile a specific batch to capture detailed memory allocations.

        Args:
            profile_batch: Which batch to profile (1 = after warmup)
            snapshot_path: Where to save the memory snapshot
        """
        super().__init__()
        self.profile_batch = profile_batch
        self.snapshot_path = snapshot_path
        self.profiling = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Start profiling right before the target batch"""
        if batch_idx == self.profile_batch and not self.profiling:
            print(f"\n{'='*70}")
            print(f"🔍 Starting memory profiling at batch {batch_idx}")
            print(f"{'='*70}")

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="all"
            )
            self.profiling = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Stop profiling right after the target batch"""
        if batch_idx == self.profile_batch and self.profiling:
            torch.cuda.synchronize()

            print(f"\n{'='*70}")
            print(f"✓ Finished profiling batch {batch_idx}")
            print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            print(f"Current memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"{'='*70}\n")

            torch.cuda.memory._record_memory_history(enabled=None)
            torch.cuda.memory._dump_snapshot(self.snapshot_path)

            self.profiling = False
            print(f"📊 Memory snapshot saved to: {self.snapshot_path}")

            # Stop training after profiling (optional)
            trainer.should_stop = True


# Memory profiling callback for ENTIRE training
class MemoryProfilerCallback(pl.Callback):
    def __init__(self, snapshot_path="stgym_train_level_memory_profile.pickle"):
        """
        Profile the ENTIRE training process.

        Args:
            snapshot_path: Where to save the memory snapshot
        """
        super().__init__()
        self.snapshot_path = snapshot_path

    def on_train_start(self, trainer, pl_module):
        """Start profiling when training begins"""
        print(f"\n{'='*70}")
        print(f"🔍 Starting memory profiling for ENTIRE training")
        print(f"{'='*70}\n")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="all",
            # max_entries=100000,
            # context="all"
        )

    def on_train_end(self, trainer, pl_module):
        """Stop profiling when training ends"""
        torch.cuda.synchronize()

        print(f"\n{'='*70}")
        print(f"✓ Training complete, stopping profiling")
        print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Current memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"{'='*70}\n")

        torch.cuda.memory._record_memory_history(enabled=None)
        torch.cuda.memory._dump_snapshot(self.snapshot_path)

        print(f"📊 Memory snapshot saved to: {self.snapshot_path}")

        trainer.should_stop = True


class DataTransferProfilerCallback(pl.Callback):
    """It does not work -- hooks are not entered at all"""

    def __init__(self, profile_batch=1, snapshot_path="data_transfer_profile.pickle"):
        """Profile ONLY the data transfer to GPU step"""
        super().__init__()
        self.profile_batch = profile_batch
        self.snapshot_path = snapshot_path
        self.profiling = False

    def on_before_batch_transfer(self, batch, dataloader_idx):
        print(f"[on_before_batch_transfer] Dataloader {dataloader_idx}")
        x, y = batch
        print(f"  Batch device BEFORE transfer: x={x.device}, y={y.device}")
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        print(f"[on_after_batch_transfer] Dataloader {dataloader_idx}")
        x, y = batch
        print(f"  Batch device AFTER transfer: x={x.device}, y={y.device}")
        return batch

    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     """Start profiling RIGHT BEFORE batch is moved to GPU"""
    #     print(f"\n{'='*70}")
    #     print(f"🔍 Starting profiling BEFORE data transfer")
    #     print(f"   Batch is currently on CPU, about to move to GPU...")
    #     print(f"{'='*70}\n")

    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats()

    #     torch.cuda.memory._record_memory_history(
    #         enabled="all",
    #         max_entries=100000,
    #         context="all",
    #         stacks="all"
    #     )
    #     self.profiling = True

    #     return batch

    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     """Stop profiling RIGHT AFTER batch is moved to GPU"""
    #     torch.cuda.synchronize()

    #     print(f"\n{'='*70}")
    #     print(f"✓ Finished profiling data transfer")
    #     print(f"   Batch is now on GPU")
    #     print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    #     print(f"Current memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    #     print(f"{'='*70}\n")

    #     torch.cuda.memory._record_memory_history(enabled=None)
    #     torch.cuda.memory._dump_snapshot(self.snapshot_path)

    #     self.profiling = False
    #     print(f"📊 Data transfer snapshot saved to: {self.snapshot_path}")

    #     # Stop training after profiling
    #     self._trainer.should_stop = True

    #     return batch
