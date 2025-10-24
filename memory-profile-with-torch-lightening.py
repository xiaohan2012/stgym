import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Simple model as LightningModule
class SimpleModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Memory profiling callback
class MemoryProfilerCallback(L.Callback):
    def __init__(
        self, profile_batch=1, snapshot_path="lightning_memory_profile.pickle"
    ):
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
                enabled=True,
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


def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Memory profiling only works on GPU!")
        return

    print(f"Using device: cuda")

    # Create synthetic dataset
    batch_size = 128
    num_samples = 1000

    X = torch.randn(num_samples, 1024)
    y = torch.randint(0, 10, (num_samples,))

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Dataset size: {num_samples} samples")
    print(f"Batch size: {batch_size}")

    # Create model
    model = SimpleModel()
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Create trainer with memory profiler callback
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        callbacks=[
            MemoryProfilerCallback(
                profile_batch=1,  # Profile batch 1 (batch 0 is warmup)
                snapshot_path="lightning_memory_profile.pickle",
            )
        ],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=False,  # Disable default logger for cleaner output
    )

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print("Batch 0 will be used as warmup (allocates memory)")
    print("Batch 1 will be profiled (captures detailed allocations)")
    print("=" * 70 + "\n")

    # Train (will automatically stop after profiling batch 1)
    trainer.fit(model, train_loader)

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print("Memory snapshot saved to: lightning_memory_profile.pickle")
    print("\nTo visualize:")
    print("1. Go to: https://pytorch.org/memory_viz")
    print("2. Upload: lightning_memory_profile.pickle")
    print("3. Select: 'Active Memory Timeline'")
    print("\nYou should see a detailed, colorful graph showing:")
    print("  - Layer-by-layer allocations during forward pass")
    print("  - Gradient allocations during backward pass")
    print("  - Temporary tensors created and destroyed")
    print("=" * 70)


if __name__ == "__main__":
    main()
