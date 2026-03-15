#!/usr/bin/env python3
"""
Artifact fetching script for MLflow experiment failures.
Uses SCP to retrieve error logs and configuration files from remote servers.
"""

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List


class ArtifactFetcher:
    """Fetches MLflow artifacts from remote servers using SCP."""

    def __init__(
        self,
        remote_host: str = "cyy2",
        base_path: str = "~/stgym/mlruns",
        max_workers: int = 5,
    ):
        """
        Initialize artifact fetcher.

        Args:
            remote_host: SSH hostname for remote server
            base_path: Base path to MLflow runs directory on remote server
            max_workers: Maximum number of parallel SCP operations
        """
        self.remote_host = remote_host
        self.base_path = base_path
        self.max_workers = max_workers

    def test_connectivity(self, experiment_id: str) -> bool:
        """
        Test SSH connectivity and experiment directory existence.

        Args:
            experiment_id: MLflow experiment ID

        Returns:
            True if connectivity successful and directory exists
        """
        try:
            remote_path = f"{self.base_path}/{experiment_id}"
            cmd = ["ssh", self.remote_host, f"ls -la {remote_path}"]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"SSH connection to {self.remote_host} timed out", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Connectivity test failed: {e}", file=sys.stderr)
            return False

    def check_run_artifacts(self, experiment_id: str, run_id: str) -> Dict[str, bool]:
        """
        Check which artifacts are available for a specific run.

        Args:
            experiment_id: MLflow experiment ID
            run_id: MLflow run ID

        Returns:
            Dict indicating which artifact files are available
        """
        artifacts_path = f"{self.base_path}/{experiment_id}/{run_id}/artifacts"
        cmd = ["ssh", self.remote_host, f"ls -la {artifacts_path}"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                return {
                    "training_error.txt": False,
                    "experiment_config.yaml": False,
                    "directory_exists": False,
                }

            file_list = result.stdout
            return {
                "training_error.txt": "training_error.txt" in file_list,
                "experiment_config.yaml": "experiment_config.yaml" in file_list,
                "directory_exists": True,
            }

        except Exception:
            return {
                "training_error.txt": False,
                "experiment_config.yaml": False,
                "directory_exists": False,
            }

    def fetch_single_artifact(
        self, experiment_id: str, run_id: str, artifact_name: str, local_path: str
    ) -> Dict:
        """
        Fetch a single artifact file using SCP.

        Args:
            experiment_id: MLflow experiment ID
            run_id: MLflow run ID
            artifact_name: Name of artifact file to fetch
            local_path: Local path to save the file

        Returns:
            Dict with fetch results and metadata
        """
        start_time = time.time()

        remote_path = f"{self.remote_host}:{self.base_path}/{experiment_id}/{run_id}/artifacts/{artifact_name}"

        # Ensure local directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            cmd = ["scp", remote_path, local_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            success = result.returncode == 0
            file_size = (
                Path(local_path).stat().st_size
                if success and Path(local_path).exists()
                else 0
            )

            return {
                "run_id": run_id,
                "artifact_name": artifact_name,
                "success": success,
                "local_path": local_path,
                "file_size": file_size,
                "duration": time.time() - start_time,
                "error_message": result.stderr if not success else None,
            }

        except subprocess.TimeoutExpired:
            return {
                "run_id": run_id,
                "artifact_name": artifact_name,
                "success": False,
                "local_path": local_path,
                "file_size": 0,
                "duration": time.time() - start_time,
                "error_message": "SCP operation timed out",
            }
        except Exception as e:
            return {
                "run_id": run_id,
                "artifact_name": artifact_name,
                "success": False,
                "local_path": local_path,
                "file_size": 0,
                "duration": time.time() - start_time,
                "error_message": str(e),
            }

    def fetch_batch_artifacts(
        self, experiment_id: str, run_ids: List[str], local_base_dir: str
    ) -> Dict:
        """
        Fetch artifacts for multiple runs in parallel.

        Args:
            experiment_id: MLflow experiment ID
            run_ids: List of run IDs to fetch artifacts for
            local_base_dir: Base directory for local storage

        Returns:
            Dict with batch fetch results and statistics
        """
        # Create timestamped working directory
        timestamp = int(time.time())
        work_dir = Path(local_base_dir) / f"mlflow_analysis_{timestamp}"
        errors_dir = work_dir / "errors"
        configs_dir = work_dir / "configs"

        # Ensure directories exist
        errors_dir.mkdir(parents=True, exist_ok=True)
        configs_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "experiment_id": experiment_id,
            "work_directory": str(work_dir),
            "total_runs": len(run_ids),
            "fetched_errors": 0,
            "fetched_configs": 0,
            "failed_fetches": 0,
            "fetch_details": [],
            "summary": {},
            "start_time": time.time(),
        }

        # Create list of fetch tasks
        fetch_tasks = []
        for i, run_id in enumerate(run_ids):
            # Error file task
            error_local_path = errors_dir / f"error_{i+1}.txt"
            fetch_tasks.append(
                (experiment_id, run_id, "training_error.txt", str(error_local_path))
            )

            # Config file task
            config_local_path = configs_dir / f"config_{i+1}.yaml"
            fetch_tasks.append(
                (
                    experiment_id,
                    run_id,
                    "experiment_config.yaml",
                    str(config_local_path),
                )
            )

        # Execute fetches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.fetch_single_artifact, *task): task
                for task in fetch_tasks
            }

            # Process completed tasks
            completed = 0
            total_tasks = len(fetch_tasks)

            for future in as_completed(future_to_task):
                task = future_to_task[future]

                try:
                    fetch_result = future.result()
                    results["fetch_details"].append(fetch_result)

                    if fetch_result["success"]:
                        if fetch_result["artifact_name"] == "training_error.txt":
                            results["fetched_errors"] += 1
                        elif fetch_result["artifact_name"] == "experiment_config.yaml":
                            results["fetched_configs"] += 1
                    else:
                        results["failed_fetches"] += 1

                except Exception as e:
                    results["failed_fetches"] += 1
                    print(f"Task failed for {task}: {e}", file=sys.stderr)

                completed += 1
                if completed % 10 == 0:
                    print(
                        f"Progress: {completed}/{total_tasks} artifacts processed",
                        file=sys.stderr,
                    )

        # Generate summary statistics
        total_time = time.time() - results["start_time"]
        successful_fetches = results["fetched_errors"] + results["fetched_configs"]

        results["summary"] = {
            "total_duration": total_time,
            "success_rate": successful_fetches / total_tasks if total_tasks > 0 else 0,
            "avg_fetch_time": (
                sum(d["duration"] for d in results["fetch_details"])
                / len(results["fetch_details"])
                if results["fetch_details"]
                else 0
            ),
            "total_data_size": sum(d["file_size"] for d in results["fetch_details"]),
            "errors_success_rate": (
                results["fetched_errors"] / results["total_runs"]
                if results["total_runs"] > 0
                else 0
            ),
            "configs_success_rate": (
                results["fetched_configs"] / results["total_runs"]
                if results["total_runs"] > 0
                else 0
            ),
        }

        return results

    def generate_report(self, fetch_results: Dict) -> str:
        """
        Generate a human-readable report from fetch results.

        Args:
            fetch_results: Results from fetch_batch_artifacts

        Returns:
            Formatted report string
        """
        summary = fetch_results["summary"]

        report_lines = [
            "# Artifact Fetch Results Report",
            "",
            f"**Experiment ID**: {fetch_results['experiment_id']}",
            f"**Remote server**: {self.remote_host}",
            f"**Working directory**: {fetch_results['work_directory']}",
            f"**Total runs processed**: {fetch_results['total_runs']}",
            "",
            "## Fetch Statistics",
            f"- **Error logs retrieved**: {fetch_results['fetched_errors']}/{fetch_results['total_runs']} ({summary['errors_success_rate']:.1%})",
            f"- **Config files retrieved**: {fetch_results['fetched_configs']}/{fetch_results['total_runs']} ({summary['configs_success_rate']:.1%})",
            f"- **Failed fetches**: {fetch_results['failed_fetches']}",
            f"- **Total processing time**: {summary['total_duration']:.1f} seconds",
            f"- **Average fetch time**: {summary['avg_fetch_time']:.2f} seconds",
            f"- **Total data retrieved**: {summary['total_data_size']:,} bytes",
            "",
        ]

        if fetch_results["failed_fetches"] > 0:
            failed_details = [
                d for d in fetch_results["fetch_details"] if not d["success"]
            ]

            report_lines.extend(["## Failed Fetches", ""])

            # Group by error type
            error_types = {}
            for detail in failed_details[:10]:  # Show first 10 failures
                error_msg = detail.get("error_message", "Unknown error")
                if error_msg not in error_types:
                    error_types[error_msg] = []
                error_types[error_msg].append(detail["run_id"])

            for error_msg, run_ids in error_types.items():
                report_lines.extend(
                    [
                        f"**Error**: {error_msg}",
                        f"**Affected runs**: {len(run_ids)} runs",
                        "",
                    ]
                )

        return "\n".join(report_lines)


def main():
    """Command-line interface for artifact fetching."""
    if len(sys.argv) < 4:
        print(
            "Usage: python fetch_artifacts.py <experiment_id> <run_ids_file> <output_dir> [remote_host] [base_path]",
            file=sys.stderr,
        )
        print(
            "  run_ids_file: JSON file with list of run IDs to fetch", file=sys.stderr
        )
        sys.exit(1)

    experiment_id = sys.argv[1]
    run_ids_file = sys.argv[2]
    output_dir = sys.argv[3]
    remote_host = sys.argv[4] if len(sys.argv) > 4 else "cyy2"
    base_path = sys.argv[5] if len(sys.argv) > 5 else "~/stgym/mlruns"

    try:
        # Load run IDs
        with open(run_ids_file) as f:
            run_ids = json.load(f)

        if not isinstance(run_ids, list):
            print("run_ids_file must contain a JSON list of run IDs", file=sys.stderr)
            sys.exit(1)

        # Initialize fetcher
        fetcher = ArtifactFetcher(remote_host=remote_host, base_path=base_path)

        # Test connectivity
        print(f"Testing connectivity to {remote_host}...", file=sys.stderr)
        if not fetcher.test_connectivity(experiment_id):
            print(
                f"Failed to connect to {remote_host} or experiment directory not found",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            f"Successfully connected. Fetching artifacts for {len(run_ids)} runs...",
            file=sys.stderr,
        )

        # Fetch artifacts
        results = fetcher.fetch_batch_artifacts(experiment_id, run_ids, output_dir)

        # Output JSON results
        print(json.dumps(results, default=str, indent=2))

        # Output human-readable report to stderr
        report = fetcher.generate_report(results)
        print(report, file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
