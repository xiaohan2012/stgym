import torch

# Try to allocate close to 24GB
device = torch.device("cuda:0")
try:
    # Allocate ~20GB tensor (leaving some for overhead)
    tensor_size = 20 * 1024 * 1024 * 1024 // 4  # 20GB in float32
    x = torch.randn(tensor_size, device=device, dtype=torch.float32)
    print(f"Successfully allocated {x.element_size() * x.nelement() / 1e9:.2f} GB")
    del x
    torch.cuda.empty_cache()
except RuntimeError as e:
    print(f"Failed: {e}")

import os

# Check CUDA libraries PyTorch is using
print(torch.cuda.is_available())
print(torch.version.cuda)

# See what CUDA library is loaded
os.system(
    'ldd $(python -c \'import torch; print(torch.__file__.replace("__init__.py", "lib/libtorch_cuda.so"))\')'
)


import time

device = torch.device("cuda:0")


# Matrix multiplication benchmark
def benchmark_matmul(size=10000, iterations=100):
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(10):
        torch.matmul(A, B)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    end = time.time()

    elapsed = (end - start) / iterations
    tflops = (2 * size**3) / (elapsed * 1e12)
    return elapsed, tflops


elapsed, tflops = benchmark_matmul()
print(f"Time per matmul: {elapsed*1000:.2f} ms")
print(f"Performance: {tflops:.2f} TFLOPS")
