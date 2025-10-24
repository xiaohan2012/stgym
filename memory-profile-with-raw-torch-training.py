import torch
import torch.nn as nn
import torch.optim as optim


# Simple model for testing
class SimpleModel(nn.Module):
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

    def forward(self, x):
        return self.layers(x)


# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("WARNING: CUDA not available. Memory profiling only works on GPU!")
    exit(1)

model = SimpleModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Create dummy data
batch_size = 128
data = torch.randn(batch_size, 1024, device=device)
target = torch.randint(0, 10, (batch_size,), device=device)

print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
print(f"Input shape: {data.shape}")
print(f"Target shape: {target.shape}")

print("\n" + "=" * 70)
print("WARMUP: Running one step to allocate memory")
print("=" * 70)

# Warmup step - allocate all the memory first
model.train()
optimizer.zero_grad()
output = model(data)
loss = criterion(output, target)
loss.backward()
optimizer.step()

print(f"Warmup complete. Loss: {loss.item():.4f}")
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

print("\n" + "=" * 70)
print("PROFILING: Recording detailed memory timeline for ONE step")
print("=" * 70)

# Clear cache and reset stats
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Start profiling
print("Starting memory profiling...")
torch.cuda.memory._record_memory_history(
    enabled=True,
)

# Profile this single training step
optimizer.zero_grad()
output = model(data)
loss = criterion(output, target)
loss.backward()
optimizer.step()

# Stop profiling
torch.cuda.synchronize()
torch.cuda.memory._record_memory_history(enabled=None)

print(f"Profiling complete. Loss: {loss.item():.4f}")
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Current memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Save snapshot
snapshot_file = "memory_profile.pickle"
torch.cuda.memory._dump_snapshot(snapshot_file)

print("\n" + "=" * 70)
print("SUCCESS!")
print("=" * 70)
print(f"Memory snapshot saved to: {snapshot_file}")
print("\nTo visualize:")
print("1. Go to: https://pytorch.org/memory_viz")
print("2. Upload: memory_profile.pickle")
print("3. Select: 'Active Memory Timeline'")
print("\nYou should see a detailed, colorful graph showing:")
print("  - Layer-by-layer allocations during forward pass")
print("  - Gradient allocations during backward pass")
print("  - Temporary tensors created and destroyed")
print("=" * 70)
