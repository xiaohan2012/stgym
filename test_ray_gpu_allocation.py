#!/usr/bin/env python
"""
Test script to demonstrate Ray GPU allocation and device visibility.

This script shows how ray.get_gpu_ids() works inside Ray workers and how
CUDA_VISIBLE_DEVICES affects PyTorch device visibility.
"""

import os
from typing import List

import ray
import torch


@ray.remote(num_gpus=1.0)
def test_gpu_allocation_full(worker_id: int) -> dict:
    """
    Ray worker that uses 1 full GPU to test device allocation.

    Args:
        worker_id: Unique identifier for this worker

    Returns:
        Dictionary with device information from this worker
    """
    result = {"worker_id": worker_id, "gpu_allocation": "1.0 GPU"}

    # Get Ray-assigned GPU IDs
    ray_gpu_ids = ray.get_gpu_ids()
    result["ray_gpu_ids"] = ray_gpu_ids
    result["ray_gpu_count"] = len(ray_gpu_ids)

    # Check original CUDA device count (before CUDA_VISIBLE_DEVICES)
    original_device_count = (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    )
    result["original_cuda_device_count"] = original_device_count

    # Apply the same device logic as our fixed run_exp()
    if torch.cuda.is_available() and ray_gpu_ids:
        # Use the first assigned GPU
        device_id = ray_gpu_ids[0]

        # there is no need to assign the environment variable
        # ray handles it already
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        print(os.environ["CUDA_VISIBLE_DEVICES"])

        # Check device count after restriction
        restricted_device_count = torch.cuda.device_count()
        result["restricted_cuda_device_count"] = restricted_device_count
        result["cuda_visible_devices"] = str(device_id)

        # Test PyTorch device
        if restricted_device_count > 0:
            test_tensor = torch.tensor([1.0]).cuda()
            result["pytorch_device"] = str(test_tensor.device)
        else:
            result["pytorch_device"] = "no_cuda_available"
    else:
        result["restricted_cuda_device_count"] = 0
        result["cuda_visible_devices"] = "not_set"
        result["pytorch_device"] = "cpu"

    return result


@ray.remote(num_gpus=0.5)
def test_gpu_allocation_half(worker_id: int) -> dict:
    """
    Ray worker that uses 0.5 GPU to test device allocation.

    Args:
        worker_id: Unique identifier for this worker

    Returns:
        Dictionary with device information from this worker
    """
    result = {"worker_id": worker_id, "gpu_allocation": "0.5 GPU"}

    # Get Ray-assigned GPU IDs
    ray_gpu_ids = ray.get_gpu_ids()
    result["ray_gpu_ids"] = ray_gpu_ids
    result["ray_gpu_count"] = len(ray_gpu_ids)

    # Check original CUDA device count (before CUDA_VISIBLE_DEVICES)
    original_device_count = (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    )
    result["original_cuda_device_count"] = original_device_count

    # Apply the same device logic as our fixed run_exp()
    if torch.cuda.is_available() and ray_gpu_ids:
        # Use the first assigned GPU
        device_id = ray_gpu_ids[0]
        print(os.environ["CUDA_VISIBLE_DEVICES"])

        # there is no need to assign the environment variable
        # ray handles it already
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        # Check device count after restriction
        restricted_device_count = torch.cuda.device_count()
        result["restricted_cuda_device_count"] = restricted_device_count
        result["cuda_visible_devices"] = str(device_id)

        # Test PyTorch device
        if restricted_device_count > 0:
            test_tensor = torch.tensor([1.0]).cuda()
            result["pytorch_device"] = str(test_tensor.device)
        else:
            result["pytorch_device"] = "no_cuda_available"
    else:
        result["restricted_cuda_device_count"] = 0
        result["cuda_visible_devices"] = "not_set"
        result["pytorch_device"] = "cpu"

    return result


def print_worker_results(results: List[dict]) -> None:
    """Print worker results in a formatted way."""
    print("\n" + "=" * 80)
    print("🔍 RAY GPU ALLOCATION TEST RESULTS")
    print("=" * 80)

    for result in results:
        worker_id = result["worker_id"]
        gpu_allocation = result["gpu_allocation"]
        print(f"\n📋 Worker {worker_id} ({gpu_allocation}):")
        print(f"   Ray GPU IDs: {result['ray_gpu_ids']}")
        print(f"   Ray GPU count: {result['ray_gpu_count']}")
        print(f"   Original CUDA devices: {result['original_cuda_device_count']}")
        print(f"   Restricted CUDA devices: {result['restricted_cuda_device_count']}")
        print(f"   CUDA_VISIBLE_DEVICES: {result['cuda_visible_devices']}")
        print(f"   PyTorch device: {result['pytorch_device']}")

        # Verification
        if result["ray_gpu_count"] == 1 and result["restricted_cuda_device_count"] == 1:
            print(f"   ✅ SUCCESS: Each worker sees exactly 1 GPU")
        else:
            print(
                f"   ❌ ISSUE: Expected 1 GPU, got Ray={result['ray_gpu_count']}, PyTorch={result['restricted_cuda_device_count']}"
            )


def main():
    """Main function to test Ray GPU allocation."""
    print("🚀 Starting Ray GPU allocation test...")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available - this test requires GPUs")
        return

    total_gpus = torch.cuda.device_count()
    print(f"💻 Total GPUs available: {total_gpus}")

    if total_gpus < 2:
        print("⚠️  Warning: This test is designed for systems with 2+ GPUs")

    # Initialize Ray with GPU support
    ray.init(num_gpus=total_gpus)
    print(f"🎯 Ray initialized with {total_gpus} GPUs")

    try:
        # Launch workers with different GPU allocations
        print("\n🔄 Launching Ray workers...")

        # Test with full GPU allocation (1.0 GPU each)
        full_gpu_futures = []
        for i in range(2):  # Launch 2 workers with 1.0 GPU each
            future = test_gpu_allocation_full.remote(worker_id=i + 1)
            full_gpu_futures.append(future)

        # Test with half GPU allocation (0.5 GPU each)
        half_gpu_futures = []
        for i in range(4):  # Launch 4 workers with 0.5 GPU each
            future = test_gpu_allocation_half.remote(worker_id=i + 3)
            half_gpu_futures.append(future)

        print(f"   Launched 2 workers with 1.0 GPU each")
        print(f"   Launched 4 workers with 0.5 GPU each")

        # Wait for results
        print("\n⏳ Waiting for worker results...")
        full_gpu_results = ray.get(full_gpu_futures)
        half_gpu_results = ray.get(half_gpu_futures)

        # Print results
        all_results = full_gpu_results + half_gpu_results
        print_worker_results(all_results)

        # Summary analysis
        print(f"\n📊 Summary Analysis:")
        success_count = sum(
            1
            for r in all_results
            if r["ray_gpu_count"] == 1 and r["restricted_cuda_device_count"] == 1
        )
        total_count = len(all_results)
        print(f"   Workers with correct GPU allocation: {success_count}/{total_count}")

        # Check GPU distribution
        assigned_gpus = set()
        for result in all_results:
            if result["ray_gpu_ids"]:
                assigned_gpus.update(result["ray_gpu_ids"])

        print(f"   Unique GPUs used: {sorted(assigned_gpus)}")
        print(f"   GPU utilization: {len(assigned_gpus)}/{total_gpus} GPUs used")

        if success_count == total_count:
            print(f"\n🎉 SUCCESS: All workers correctly see exactly 1 GPU device!")
        else:
            print(f"\n❌ ISSUES: Some workers don't see exactly 1 GPU device")

    finally:
        ray.shutdown()
        print(f"\n🏁 Ray shutdown complete")


if __name__ == "__main__":
    main()
