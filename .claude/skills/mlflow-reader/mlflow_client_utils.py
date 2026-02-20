#!/usr/bin/env python
"""
MLflow Client Utilities for Claude Code Skill

This module provides reusable utilities for interacting with MLflow tracking servers.
Designed to support the mlflow-reader Claude Code skill with common operations.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import Experiment, Run


class MLflowReader:
    """MLflow client wrapper with utilities for reading tracking data."""

    def __init__(self, tracking_uri: str = "http://127.0.0.1:5000"):
        """Initialize MLflow client with tracking URI.

        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    @staticmethod
    def parse_mlflow_url(url: str) -> Tuple[str, str, str]:
        """Parse MLflow URL to extract components.

        Args:
            url: MLflow URL like http://host/#/experiments/123/runs/abc123

        Returns:
            Tuple of (tracking_uri, experiment_id, run_id)

        Raises:
            ValueError: If URL format is invalid
        """
        parsed = urlparse(url)
        tracking_uri = f"{parsed.scheme}://{parsed.netloc}"

        # Extract experiment ID and run ID from fragment
        fragment = parsed.fragment
        match = re.search(r"/experiments/(\d+)/runs/([a-f0-9]+)", fragment)

        if not match:
            raise ValueError(f"Invalid MLflow URL format: {url}")

        experiment_id = match.group(1)
        run_id = match.group(2)

        return tracking_uri, experiment_id, run_id

    def list_experiments(self, view_type: str = "ACTIVE_ONLY") -> List[Experiment]:
        """List all experiments on the tracking server.

        Args:
            view_type: "ACTIVE_ONLY", "DELETED_ONLY", or "ALL"

        Returns:
            List of Experiment objects
        """
        return self.client.search_experiments(view_type=view_type)

    def get_experiment(
        self, experiment_name_or_id: Union[str, int]
    ) -> Optional[Experiment]:
        """Get experiment by name or ID.

        Args:
            experiment_name_or_id: Experiment name or numeric ID

        Returns:
            Experiment object or None if not found
        """
        try:
            if (
                isinstance(experiment_name_or_id, str)
                and not experiment_name_or_id.isdigit()
            ):
                return self.client.get_experiment_by_name(experiment_name_or_id)
            else:
                exp_id = str(experiment_name_or_id)
                return self.client.get_experiment(exp_id)
        except Exception:
            return None

    def search_runs(
        self,
        experiment_ids: List[str],
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 1000,
    ) -> List[Run]:
        """Search runs across experiments.

        Args:
            experiment_ids: List of experiment IDs to search
            filter_string: MLflow filter string (e.g., "metrics.accuracy > 0.8")
            order_by: List of order criteria (e.g., ["metrics.accuracy DESC"])
            max_results: Maximum number of runs to return

        Returns:
            List of Run objects
        """
        return self.client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            order_by=order_by or [],
            max_results=max_results,
        )

    def get_runs_by_status(
        self,
        experiment_ids: List[str],
        status: str,
        order_by: Optional[List[str]] = None,
        max_results: int = 1000,
    ) -> List[Run]:
        """Get runs filtered by status.

        Args:
            experiment_ids: List of experiment IDs to search
            status: Run status (FINISHED, FAILED, RUNNING, SCHEDULED, KILLED)
            order_by: List of order criteria
            max_results: Maximum number of runs to return

        Returns:
            List of Run objects with the specified status
        """
        filter_string = f"attributes.status = '{status}'"
        return self.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            order_by=order_by,
            max_results=max_results,
        )

    def get_failed_runs(
        self,
        experiment_ids: List[str],
        order_by: Optional[List[str]] = None,
        max_results: int = 1000,
    ) -> List[Run]:
        """Get all failed runs from experiments.

        Args:
            experiment_ids: List of experiment IDs to search
            order_by: List of order criteria (default: newest first)
            max_results: Maximum number of runs to return

        Returns:
            List of failed Run objects
        """
        order_by = order_by or ["attributes.start_time DESC"]
        return self.get_runs_by_status(
            experiment_ids=experiment_ids,
            status="FAILED",
            order_by=order_by,
            max_results=max_results,
        )

    def get_successful_runs(
        self,
        experiment_ids: List[str],
        order_by: Optional[List[str]] = None,
        max_results: int = 1000,
    ) -> List[Run]:
        """Get all successfully finished runs from experiments.

        Args:
            experiment_ids: List of experiment IDs to search
            order_by: List of order criteria (default: newest first)
            max_results: Maximum number of runs to return

        Returns:
            List of successfully finished Run objects
        """
        order_by = order_by or ["attributes.start_time DESC"]
        return self.get_runs_by_status(
            experiment_ids=experiment_ids,
            status="FINISHED",
            order_by=order_by,
            max_results=max_results,
        )

    def get_run_data(self, run_id: str, include_error: bool = False) -> Dict[str, Any]:
        """Get complete run data including params, metrics, and metadata.

        Args:
            run_id: MLflow run ID
            include_error: Whether to include training error content if available

        Returns:
            Dictionary with run data
        """
        run = self.client.get_run(run_id)

        # Calculate duration if both start and end time are available
        duration_ms = None
        if run.info.start_time and run.info.end_time:
            duration_ms = run.info.end_time - run.info.start_time

        run_data = {
            "run_id": run_id,
            "tracking_uri": self.tracking_uri,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "duration_ms": duration_ms,
            "lifecycle_stage": run.info.lifecycle_stage,
            "user_id": run.info.user_id,
            "parameters": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
            "artifact_uri": run.info.artifact_uri,
            "artifacts": self.list_artifacts(run_id),
        }

        # Include training error if requested and run failed
        if include_error and run.info.status == "FAILED":
            training_error = self.get_training_error(run_id)
            run_data["training_error"] = training_error

        return run_data

    def get_run_from_url(self, url: str) -> Dict[str, Any]:
        """Get run data from MLflow URL.

        Args:
            url: MLflow run URL

        Returns:
            Dictionary with run data
        """
        tracking_uri, experiment_id, run_id = self.parse_mlflow_url(url)

        # Create new client if tracking URI differs
        if tracking_uri != self.tracking_uri:
            self.tracking_uri
            self.__init__(tracking_uri)

        return self.get_run_data(run_id)

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs with their parameters and metrics.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            DataFrame with runs as rows, params/metrics as columns
        """
        run_data = []

        for run_id in run_ids:
            try:
                data = self.get_run_data(run_id)
                row = {"run_id": run_id, "status": data["status"]}
                row.update({f"param_{k}": v for k, v in data["parameters"].items()})
                row.update({f"metric_{k}": v for k, v in data["metrics"].items()})
                run_data.append(row)
            except Exception as e:
                print(f"Warning: Could not retrieve data for run {run_id}: {e}")
                continue

        return pd.DataFrame(run_data)

    def export_experiment_data(
        self,
        experiment_name_or_id: Union[str, int],
        format: str = "json",
        output_path: Optional[Path] = None,
    ) -> Union[str, Path]:
        """Export experiment data to file or return as string.

        Args:
            experiment_name_or_id: Experiment name or ID
            format: Export format ("json", "csv")
            output_path: Optional file path to save data

        Returns:
            Formatted string or path to saved file
        """
        experiment = self.get_experiment(experiment_name_or_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_name_or_id}")

        runs = self.search_runs([experiment.experiment_id])

        # Prepare data
        export_data = []
        for run in runs:
            run_data = self.get_run_data(run.info.run_id)
            export_data.append(run_data)

        if format.lower() == "json":
            output = json.dumps(export_data, indent=2, default=str)
            if output_path:
                output_path.write_text(output)
                return output_path
            return output

        elif format.lower() == "csv":
            # Flatten data for CSV
            flattened = []
            for run_data in export_data:
                row = {
                    "run_id": run_data["run_id"],
                    "experiment_id": run_data["experiment_id"],
                    "status": run_data["status"],
                    "start_time": run_data["start_time"],
                    "end_time": run_data["end_time"],
                }
                # Add parameters with param_ prefix
                for k, v in run_data["parameters"].items():
                    row[f"param_{k}"] = v
                # Add metrics with metric_ prefix
                for k, v in run_data["metrics"].items():
                    row[f"metric_{k}"] = v
                flattened.append(row)

            df = pd.DataFrame(flattened)
            if output_path:
                df.to_csv(output_path, index=False)
                return output_path
            return df.to_csv(index=False)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_metric_history(self, run_id: str, metric_key: str) -> List[Dict]:
        """Get metric history for a run.

        Args:
            run_id: MLflow run ID
            metric_key: Metric name

        Returns:
            List of metric history points with timestamp, step, value
        """
        history = self.client.get_metric_history(run_id, metric_key)
        return [
            {"timestamp": point.timestamp, "step": point.step, "value": point.value}
            for point in history
        ]

    def get_training_error(self, run_id: str) -> Optional[str]:
        """Get training error content from training_error.txt artifact.

        Args:
            run_id: MLflow run ID

        Returns:
            Error text content or None if file doesn't exist
        """
        try:
            run = self.client.get_run(run_id)
            artifact_uri = run.info.artifact_uri

            # Convert artifact URI to local path
            if artifact_uri.startswith("file://"):
                artifact_path = Path(artifact_uri[7:])  # Remove "file://" prefix
            else:
                artifact_path = Path(artifact_uri)

            error_file_path = artifact_path / "training_error.txt"

            if error_file_path.exists():
                return error_file_path.read_text()
            else:
                return None

        except Exception as e:
            print(f"Error reading training_error.txt for run {run_id}: {e}")
            return None

    def read_artifact_file(self, run_id: str, artifact_path: str) -> Optional[str]:
        """Read any artifact file content from local filesystem.

        Args:
            run_id: MLflow run ID
            artifact_path: Path to artifact file relative to run's artifact directory

        Returns:
            File content as string or None if file doesn't exist
        """
        try:
            run = self.client.get_run(run_id)
            artifact_uri = run.info.artifact_uri

            # Convert artifact URI to local path
            if artifact_uri.startswith("file://"):
                base_path = Path(artifact_uri[7:])  # Remove "file://" prefix
            else:
                base_path = Path(artifact_uri)

            full_path = base_path / artifact_path

            if full_path.exists():
                return full_path.read_text()
            else:
                return None

        except Exception as e:
            print(f"Error reading artifact {artifact_path} for run {run_id}: {e}")
            return None

    def list_artifacts(self, run_id: str, path: str = "") -> List[Dict]:
        """List artifacts for a run.

        Args:
            run_id: MLflow run ID
            path: Artifact path (empty for root)

        Returns:
            List of artifact info dictionaries
        """
        artifacts = self.client.list_artifacts(run_id, path)
        return [
            {
                "path": artifact.path,
                "is_dir": artifact.is_dir,
                "file_size": artifact.file_size,
            }
            for artifact in artifacts
        ]


def create_mlflow_reader(tracking_uri: Optional[str] = None) -> MLflowReader:
    """Factory function to create MLflowReader instance.

    Args:
        tracking_uri: Optional tracking URI (uses default if not provided)

    Returns:
        MLflowReader instance
    """
    uri = tracking_uri or "http://127.0.0.1:5000"
    return MLflowReader(uri)


def get_failed_runs_with_errors(
    reader: MLflowReader, experiment_ids: List[str], max_results: int = 100
) -> List[Dict[str, Any]]:
    """Get failed runs with their error information.

    Args:
        reader: MLflowReader instance
        experiment_ids: List of experiment IDs to search
        max_results: Maximum number of failed runs to return

    Returns:
        List of run data dictionaries with error information included
    """
    failed_runs = reader.get_failed_runs(experiment_ids, max_results=max_results)
    runs_with_errors = []

    for run in failed_runs:
        run_data = reader.get_run_data(run.info.run_id, include_error=True)
        runs_with_errors.append(run_data)

    return runs_with_errors


def format_run_summary(run_data: Dict[str, Any]) -> str:
    """Format run data as human-readable summary.

    Args:
        run_data: Run data dictionary from get_run_data()

    Returns:
        Formatted summary string
    """
    lines = [
        f"Run ID: {run_data['run_id']}",
        f"Status: {run_data['status']}",
        f"Experiment: {run_data['experiment_id']}",
    ]

    if run_data.get("duration_ms"):
        duration = run_data["duration_ms"] / 1000
        lines.append(f"Duration: {duration:.1f} seconds")

    if run_data.get("tags"):
        lines.append("\nTags:")
        for key, value in run_data["tags"].items():
            lines.append(f"  {key}: {value}")

    if run_data.get("parameters"):
        lines.append("\nParameters:")
        for key, value in run_data["parameters"].items():
            lines.append(f"  {key}: {value}")

    if run_data.get("metrics"):
        lines.append("\nMetrics:")
        for key, value in run_data["metrics"].items():
            lines.append(f"  {key}: {value}")

    if run_data.get("training_error"):
        lines.append(f"\nTraining Error:")
        error_lines = run_data["training_error"].strip().split("\n")
        for line in error_lines[:10]:  # Show first 10 lines
            lines.append(f"  {line}")
        if len(error_lines) > 10:
            lines.append(f"  ... ({len(error_lines) - 10} more lines)")

    return "\n".join(lines)
