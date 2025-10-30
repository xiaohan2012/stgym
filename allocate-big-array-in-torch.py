import gc
import logging

import torch

logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_memory_stats():
    """Print current CUDA memory statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # Convert to GB
        logger.info(
            f"CUDA memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        )
    else:
        logger.warning("CUDA is not available")


def allocate_cuda_memory(size_gb=40, device="cuda:0"):
    """
    Allocate a specific amount of memory on CUDA device.

    Args:
        size_gb: Amount of memory to allocate in GB
        device: CUDA device to use
    """
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Cannot allocate GPU memory.")
        return None

    try:
        # Calculate number of float32 elements needed
        # 1 float32 = 4 bytes
        bytes_per_gb = 1024 * 1024 * 1024
        bytes_needed = size_gb * bytes_per_gb
        num_elements = bytes_needed // 4  # 4 bytes per float32

        logger.info(f"Attempting to allocate {size_gb} GB on {device}")
        logger.info(f"This requires {num_elements:,} float32 elements")

        # Print initial memory stats
        logger.info("Initial memory state:")
        print_memory_stats()

        # Allocate the tensor
        logger.info(f"Allocating tensor...")
        tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)

        # Print memory stats after allocation
        logger.info("Memory state after allocation:")
        print_memory_stats()

        logger.info(f"Successfully allocated {size_gb} GB tensor")
        logger.info(f"Tensor shape: {tensor.shape}")
        logger.info(f"Tensor dtype: {tensor.dtype}")
        logger.info(f"Tensor device: {tensor.device}")

        return tensor

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"Out of memory error: {e}")
        logger.error(f"Failed to allocate {size_gb} GB")

        # Try to clear cache and print available memory
        torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.is_available():
            total_memory = (
                torch.cuda.get_device_properties(device).total_memory / 1024**3
            )
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            available = total_memory - allocated
            logger.info(f"Total GPU memory: {total_memory:.2f} GB")
            logger.info(f"Currently allocated: {allocated:.2f} GB")
            logger.info(f"Approximately available: {available:.2f} GB")

        return None

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


def allocate_multiple_tensors(sizes_gb, device="cuda:0"):
    """
    Allocate multiple tensors to reach target memory usage.
    Useful if you want to allocate memory in chunks.

    Args:
        sizes_gb: List of sizes in GB for each tensor
        device: CUDA device to use
    """
    tensors = []
    total_allocated = 0

    for i, size in enumerate(sizes_gb):
        logger.info(f"\n--- Allocating tensor {i+1} ({size} GB) ---")
        tensor = allocate_cuda_memory(size, device)
        if tensor is not None:
            tensors.append(tensor)
            total_allocated += size
        else:
            logger.warning(f"Failed to allocate tensor {i+1}")
            break

    logger.info(
        f"\nTotal successfully allocated: {total_allocated} GB across {len(tensors)} tensors"
    )
    return tensors


def main():
    GB = 22
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Exiting.")
        return

    # Print GPU information
    device = torch.device("cuda:0")
    gpu_properties = torch.cuda.get_device_properties(device)
    total_memory_gb = gpu_properties.total_memory / 1024**3

    logger.info(f"GPU Device: {gpu_properties.name}")
    logger.info(f"Total GPU Memory: {total_memory_gb:.2f} GB")

    # Method 1: Try to allocate 40GB in a single tensor
    logger.info("\n" + "=" * 50)
    logger.info(f"Method 1: Single {GB}GB allocation")
    logger.info("=" * 50)
    large_tensor = allocate_cuda_memory(GB, device="cuda:0")

    if large_tensor is not None:
        # Keep the tensor alive
        logger.info(
            "\nTensor successfully allocated. Press Ctrl+C to exit and free memory."
        )
        try:
            while True:
                pass
        except KeyboardInterrupt:
            logger.info("\nFreeing memory...")
            del large_tensor
            torch.cuda.empty_cache()
            logger.info("Memory freed.")
    else:
        # Method 2: Try to allocate in smaller chunks
        logger.info("\n" + "=" * 50)
        logger.info("Method 2: Multiple smaller allocations to reach 40GB")
        logger.info("=" * 50)

        # Try 4 x 10GB or 8 x 5GB allocations
        chunk_sizes = [10] * int(GB / 10)  # 4 x 10GB
        # Alternative: chunk_sizes = [5] * 8  # 8 x 5GB

        tensors = allocate_multiple_tensors(chunk_sizes, device="cuda:0")

        if tensors:
            logger.info(
                "\nTensors successfully allocated. Press Ctrl+C to exit and free memory."
            )
            try:
                while True:
                    pass
            except KeyboardInterrupt:
                logger.info("\nFreeing memory...")
                del tensors
                torch.cuda.empty_cache()
                logger.info("Memory freed.")


if __name__ == "__main__":
    main()
