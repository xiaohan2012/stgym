import torch

torch.cuda.memory
torch.cuda.memory._record_memory_history()
x = torch.randn(10000, 10000, device="cuda")  # Allocate ~400MB
y = torch.randn(10000, 10000, device="cuda")
z = x @ y  # Matrix multiply
torch.cuda.synchronize()
torch.cuda.memory._record_memory_history(enabled=None)
torch.cuda.memory._dump_snapshot("test.pickle")
