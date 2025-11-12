#!/usr/bin/env python3
"""Ensure all PyTorch Geometric datasets are processed.

This script loads all available datasets to trigger their processing,
ensuring that processed data files are created for all datasets.

Usage:
    python scripts/ensure_data_processed.py
    python scripts/ensure_data_processed.py --data-root ./data
    python scripts/ensure_data_processed.py --force-reprocess
"""
import argparse
import shutil
import sys
from pathlib import Path

# Add the project root to sys.path so we can import stgym
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stgym.data_loader import get_dataset_class
from stgym.data_loader.ds_info import get_all_ds_names


def main():
    parser = argparse.ArgumentParser(description="Ensure all datasets are processed")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for dataset storage (default: ./data)",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing by clearing processed cache",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Get all dataset names
    ds_name_list = get_all_ds_names()

    print(f"Processing {len(ds_name_list)} datasets...")
    print(f"Data root: {args.data_root}")
    if args.force_reprocess:
        print("Force reprocessing enabled")
    print()

    successful = []
    failed = []

    for ds_name in ds_name_list:
        try:
            if args.verbose:
                print(f"Processing {ds_name}...", end=" ")

            # Get dataset class
            dataset_cls = get_dataset_class(ds_name)

            # Clear processed cache if force reprocess is enabled
            data_path = Path(args.data_root) / ds_name
            if args.force_reprocess:
                processed_dir = data_path / "processed"
                if processed_dir.exists():
                    shutil.rmtree(processed_dir)
                    if args.verbose:
                        print("(cleared cache)", end=" ")

            # Instantiate dataset to trigger processing
            ds = dataset_cls(root=str(data_path))

            # Access length to ensure processing is complete
            dataset_size = len(ds)

            successful.append(ds_name)
            if args.verbose:
                print(f"✅ ({dataset_size} samples)")
            else:
                print(f"✅ {ds_name} ({dataset_size} samples)")

        except FileNotFoundError as e:
            failed.append((ds_name, f"Data files not found: {e}"))
            if args.verbose:
                print(f"❌ Data files not found")
            else:
                print(f"❌ {ds_name} - Data files not found")

        except Exception as e:
            failed.append((ds_name, str(e)))
            if args.verbose:
                print(f"❌ Error: {e}")
            else:
                print(f"❌ {ds_name} - Error: {e}")

    # Print summary
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Successfully processed: {len(successful)}/{len(ds_name_list)} datasets")
    if successful:
        print("✅ Successful datasets:")
        for ds_name in successful:
            print(f"   - {ds_name}")

    if failed:
        print(f"\n❌ Failed datasets: {len(failed)}")
        for ds_name, error in failed:
            print(f"   - {ds_name}: {error}")

    print(f"\nProcessed data stored in: {args.data_root}")

    # Exit with error code if any failed
    if failed:
        sys.exit(1)
    else:
        print("\n🎉 All datasets processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
