#!/usr/bin/env python3
"""
Standalone script to profile memory estimation performance.
Useful for identifying bottlenecks in estimate_memory_usage function.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from run_experiment_by_yaml import load_yaml_config
from stgym.mem_utils import (
    estimate_memory_usage,
    print_profiling_summary,
    reset_profiling_stats,
)


def main():
    """Profile memory estimation with a sample experiment configuration."""
    print("🔍 Starting memory estimation profiling...")

    # Load config from YAML using existing function
    config_path = str(
        Path(__file__).parent.parent.parent / "conf" / "adhoc" / "test.yaml"
    )
    exp_cfg = load_yaml_config(config_path)
    print(
        f"📋 Using config: {exp_cfg.task.dataset_name} dataset, {exp_cfg.task.type} task"
    )

    # Reset profiling statistics
    reset_profiling_stats()

    # Run memory estimation multiple times to see patterns
    num_iterations = 5
    print(f"\n🔄 Running {num_iterations} iterations of memory estimation...")

    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        total_memory, breakdown = estimate_memory_usage(exp_cfg, enable_profiling=True)
        print(f"💾 Memory estimate: {total_memory:.3f}GB")

    # Print cumulative profiling summary
    print_profiling_summary()

    print("\n✅ Profiling complete!")


if __name__ == "__main__":
    main()
