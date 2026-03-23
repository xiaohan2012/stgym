"""Run on cyy2 after an OOM sweep to verify all GPU slots were released."""

import torch

for i in range(torch.cuda.device_count()):
    try:
        x = torch.zeros(1024 * 1024, device=f"cuda:{i}")  # ~4 MB
        del x
        print(f"GPU {i}: OK (slot free)")
    except Exception as e:
        print(f"GPU {i}: FAILED — {e}")
