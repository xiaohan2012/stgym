#!/usr/bin/env python
"""
Test script for the MLflow Reader skill utilities.

This script validates the MLflow client utilities work with the STGym project setup.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mlflow_client_utils import MLflowReader, create_mlflow_reader, format_run_summary


def test_url_parsing():
    """Test MLflow URL parsing functionality."""
    print("=== Testing URL Parsing ===")

    test_url = "http://127.0.0.1:5001/#/experiments/885309957287214681/runs/32a2aa5d98444da8b511a80ccd68f2e3"

    try:
        tracking_uri, exp_id, run_id = MLflowReader.parse_mlflow_url(test_url)
        print(f"✅ URL parsed successfully:")
        print(f"  Tracking URI: {tracking_uri}")
        print(f"  Experiment ID: {exp_id}")
        print(f"  Run ID: {run_id}")
        return True
    except Exception as e:
        print(f"❌ URL parsing failed: {e}")
        return False


def test_client_creation():
    """Test MLflow client creation."""
    print("\n=== Testing Client Creation ===")

    try:
        reader = create_mlflow_reader()
        print(f"✅ MLflow reader created with URI: {reader.tracking_uri}")
        return True
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        return False


def test_connection(tracking_uri="http://127.0.0.1:5000"):
    """Test connection to MLflow server."""
    print(f"\n=== Testing Connection to {tracking_uri} ===")

    try:
        reader = MLflowReader(tracking_uri)
        experiments = reader.list_experiments()
        print(f"✅ Connected successfully. Found {len(experiments)} experiments")

        if experiments:
            exp = experiments[0]
            print(f"  Example experiment: {exp.name} (ID: {exp.experiment_id})")

        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"  Make sure MLflow server is running on {tracking_uri}")
        return False


def test_run_summary_formatting():
    """Test run data formatting."""
    print("\n=== Testing Run Summary Formatting ===")

    # Mock run data
    mock_run_data = {
        "run_id": "test123",
        "status": "FINISHED",
        "experiment_id": "1",
        "start_time": 1640000000000,  # Mock timestamp
        "end_time": 1640000060000,  # 60 seconds later
        "parameters": {"learning_rate": "0.001", "batch_size": "32", "epochs": "100"},
        "metrics": {"accuracy": 0.95, "loss": 0.1234, "val_accuracy": 0.93},
        "tags": {
            "task_type": "node-classification",
            "model": "GCN",
            "experiment_batch": "exp_001",
        },
        "duration_ms": 60000,
        "artifacts": [
            {"path": "model.pkl", "is_dir": False, "file_size": 1024},
            {"path": "logs/", "is_dir": True, "file_size": None},
        ],
    }

    try:
        summary = format_run_summary(mock_run_data)
        print("✅ Run summary formatted successfully:")
        print(summary)
        return True
    except Exception as e:
        print(f"❌ Run summary formatting failed: {e}")
        return False


def test_status_filtering(tracking_uri="http://127.0.0.1:5000"):
    """Test status-based run filtering."""
    print(f"\n=== Testing Status Filtering ===")

    try:
        reader = MLflowReader(tracking_uri)
        experiments = reader.list_experiments()

        if not experiments:
            print("⚠️  No experiments found, skipping status filtering test")
            return True

        exp_id = experiments[0].experiment_id
        print(f"Testing with experiment ID: {exp_id}")

        # Test different status filters
        failed_runs = reader.get_failed_runs([exp_id], max_results=5)
        successful_runs = reader.get_successful_runs([exp_id], max_results=5)
        running_runs = reader.get_runs_by_status([exp_id], "RUNNING", max_results=5)

        print(f"✅ Status filtering successful:")
        print(f"  Failed runs: {len(failed_runs)}")
        print(f"  Successful runs: {len(successful_runs)}")
        print(f"  Running runs: {len(running_runs)}")

        return True
    except Exception as e:
        print(f"❌ Status filtering test failed: {e}")
        return False


def test_error_extraction(tracking_uri="http://127.0.0.1:5000"):
    """Test error file reading functionality."""
    print(f"\n=== Testing Error Extraction ===")

    try:
        reader = MLflowReader(tracking_uri)
        experiments = reader.list_experiments()

        if not experiments:
            print("⚠️  No experiments found, skipping error extraction test")
            return True

        exp_id = experiments[0].experiment_id
        failed_runs = reader.get_failed_runs([exp_id], max_results=3)

        if not failed_runs:
            print("⚠️  No failed runs found, skipping error extraction test")
            return True

        print(f"Testing error extraction with {len(failed_runs)} failed runs")

        errors_found = 0
        for run in failed_runs:
            run_id = run.info.run_id
            error_content = reader.get_training_error(run_id)
            if error_content:
                errors_found += 1
                print(
                    f"✅ Error content found for run {run_id[:8]}... ({len(error_content)} chars)"
                )

        print(
            f"✅ Error extraction test completed. Found errors in {errors_found}/{len(failed_runs)} failed runs"
        )
        return True

    except Exception as e:
        print(f"❌ Error extraction test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing MLflow Reader Skill Utilities\n")

    tests = [
        test_url_parsing,
        test_client_creation,
        test_run_summary_formatting,
        lambda: test_connection("http://127.0.0.1:5000"),  # Default STGym URI
        lambda: test_status_filtering("http://127.0.0.1:5000"),
        lambda: test_error_extraction("http://127.0.0.1:5000"),
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)

    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests passed! MLflow Reader skill is ready.")
    else:
        print("❌ Some tests failed. Check MLflow server availability.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
