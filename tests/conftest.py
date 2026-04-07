import os

# Disable CUDA memory caching to prevent NVML errors on virtual GPU environments (#48)
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
