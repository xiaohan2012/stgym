#!/usr/bin/env python3
"""
Configuration analysis script for MLflow experiment failures.
Analyzes training configurations to identify patterns in failed experiments.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigurationAnalyzer:
    """Analyzes training configurations to identify failure patterns."""

    def __init__(self):
        """Initialize configuration pattern analysis."""
        self.key_params = [
            # Data loading parameters
            "data_loader.batch_size",
            "data_loader.graph_const",
            "data_loader.knn_k",
            "data_loader.radius_ratio",
            "data_loader.split.num_folds",
            "data_loader.split.split_index",
            # Model architecture
            "model.mp_layers.0.layer_type",
            "model.mp_layers.0.act",
            "model.mp_layers.0.dim_inner",
            "model.mp_layers.0.pooling.type",
            "model.mp_layers.0.pooling.n_clusters",
            "model.mp_layers.0.use_batchnorm",
            # Training parameters
            "train.optim.optimizer",
            "train.optim.base_lr",
            "train.optim.weight_decay",
            "train.max_epoch",
            "train.early_stopping.metric",
            # Task configuration
            "task.dataset_name",
            "task.type",
            "task.num_classes",
        ]

    def extract_nested_value(self, config: Dict, key_path: str) -> Any:
        """
        Extract value from nested dictionary using dot notation.

        Args:
            config: Configuration dictionary
            key_path: Dot-separated path (e.g., 'model.mp_layers.0.act')

        Returns:
            Value at the specified path, or None if not found
        """
        try:
            keys = key_path.split(".")
            value = config

            for key in keys:
                # Handle list indices
                if key.isdigit():
                    value = value[int(key)]
                else:
                    value = value[key]

            return value
        except (KeyError, IndexError, TypeError):
            return None

    def analyze_single_config(self, config: Dict) -> Dict:
        """
        Analyze a single configuration file.

        Args:
            config: Parsed configuration dictionary

        Returns:
            Dict with extracted parameter values
        """
        analysis = {}

        for param_path in self.key_params:
            value = self.extract_nested_value(config, param_path)
            if value is not None:
                analysis[param_path] = value

        # Special analysis for k-fold configuration
        num_folds = self.extract_nested_value(config, "data_loader.split.num_folds")
        split_index = self.extract_nested_value(config, "data_loader.split.split_index")

        if num_folds is not None and split_index is not None:
            analysis["is_kfold"] = True
            analysis["fold_ratio"] = split_index / num_folds if num_folds > 0 else None
        else:
            analysis["is_kfold"] = False

        # Detect hierarchical pooling configuration
        pooling_type = self.extract_nested_value(
            config, "model.mp_layers.0.pooling.type"
        )
        n_clusters = self.extract_nested_value(
            config, "model.mp_layers.0.pooling.n_clusters"
        )

        if pooling_type and n_clusters:
            analysis["pooling_config"] = f"{pooling_type}_{n_clusters}"

        return analysis

    def analyze_config_directory(self, config_dir: str) -> Dict:
        """
        Analyze all configuration files in a directory.

        Args:
            config_dir: Path to directory containing YAML configuration files

        Returns:
            Dict with aggregated configuration analysis
        """
        config_path = Path(config_dir)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

        results = {
            "total_configs": 0,
            "parameter_distributions": defaultdict(Counter),
            "correlation_patterns": defaultdict(list),
            "kfold_analysis": {
                "kfold_count": 0,
                "regular_split_count": 0,
                "fold_distribution": Counter(),
            },
            "common_patterns": {},
            "file_analysis": {},
        }

        # Process each configuration file
        config_files = list(config_path.glob("*.yaml")) + list(
            config_path.glob("*.yml")
        )

        for config_file in config_files:
            try:
                with open(config_file, encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)

                # Analyze this configuration
                analysis = self.analyze_single_config(config_data)

                # Store per-file analysis
                results["file_analysis"][config_file.name] = analysis
                results["total_configs"] += 1

                # Aggregate parameter distributions
                for param, value in analysis.items():
                    if param not in ["is_kfold", "fold_ratio"]:
                        results["parameter_distributions"][param][str(value)] += 1

                # K-fold specific analysis
                if analysis.get("is_kfold", False):
                    results["kfold_analysis"]["kfold_count"] += 1
                    fold_idx = self.extract_nested_value(
                        config_data, "data_loader.split.split_index"
                    )
                    if fold_idx is not None:
                        results["kfold_analysis"]["fold_distribution"][fold_idx] += 1
                else:
                    results["kfold_analysis"]["regular_split_count"] += 1

            except Exception as e:
                print(f"Error processing {config_file}: {e}", file=sys.stderr)

        # Identify common patterns
        self._identify_patterns(results)

        return results

    def _identify_patterns(self, results: Dict):
        """Identify common configuration patterns and correlations."""
        if results["total_configs"] == 0:
            return

        # Find most common values for each parameter
        common_patterns = {}
        for param, counter in results["parameter_distributions"].items():
            if counter:
                most_common = counter.most_common(3)  # Top 3 values
                common_patterns[param] = {
                    "most_common": most_common[0][0] if most_common else None,
                    "distribution": most_common,
                    "diversity": len(counter),  # How many different values
                }

        results["common_patterns"] = common_patterns

        # K-fold analysis summary
        kfold_stats = results["kfold_analysis"]
        if kfold_stats["kfold_count"] > 0:
            kfold_rate = kfold_stats["kfold_count"] / results["total_configs"]
            results["kfold_analysis"]["kfold_rate"] = kfold_rate

            # Check for specific fold indices that appear more often
            fold_dist = kfold_stats["fold_distribution"]
            if fold_dist:
                results["kfold_analysis"]["problematic_folds"] = [
                    fold
                    for fold, count in fold_dist.most_common()
                    if count
                    > kfold_stats["kfold_count"] * 0.15  # More than 15% of k-fold runs
                ]

    def correlate_with_errors(
        self, config_analysis: Dict, error_analysis: Dict
    ) -> Dict:
        """
        Correlate configuration patterns with error types.

        Args:
            config_analysis: Results from analyze_config_directory
            error_analysis: Results from error categorization

        Returns:
            Dict with correlation analysis
        """
        correlations = {
            "error_config_patterns": defaultdict(lambda: defaultdict(Counter)),
            "kfold_error_correlation": {},
            "high_risk_configs": [],
        }

        # Map file names between error and config analysis
        error_files = set(error_analysis.get("file_analysis", {}).keys())
        config_files = set(config_analysis.get("file_analysis", {}).keys())

        # Create mapping between error files and config files
        # Assuming naming convention: error_N.txt -> config_N.yaml
        for error_file in error_files:
            # Extract number from filename
            try:
                file_num = error_file.replace("error_", "").replace(".txt", "")
                config_file = f"config_{file_num}.yaml"

                if config_file in config_files:
                    error_cat = error_analysis["file_analysis"][error_file]["category"]
                    config_data = config_analysis["file_analysis"][config_file]

                    # Record correlations
                    for param, value in config_data.items():
                        if param not in ["is_kfold", "fold_ratio"]:
                            correlations["error_config_patterns"][error_cat][param][
                                str(value)
                            ] += 1

            except Exception:
                continue  # Skip files that don't follow expected naming

        # Analyze k-fold error correlation
        kfold_errors = []
        regular_errors = []

        for error_file in error_files:
            try:
                file_num = error_file.replace("error_", "").replace(".txt", "")
                config_file = f"config_{file_num}.yaml"

                if config_file in config_files:
                    error_cat = error_analysis["file_analysis"][error_file]["category"]
                    is_kfold = config_analysis["file_analysis"][config_file].get(
                        "is_kfold", False
                    )

                    if is_kfold:
                        kfold_errors.append(error_cat)
                    else:
                        regular_errors.append(error_cat)
            except Exception:
                continue

        if kfold_errors or regular_errors:
            correlations["kfold_error_correlation"] = {
                "kfold_error_types": Counter(kfold_errors).most_common(),
                "regular_error_types": Counter(regular_errors).most_common(),
                "kfold_error_rate": (
                    len(kfold_errors) / (len(kfold_errors) + len(regular_errors))
                    if (len(kfold_errors) + len(regular_errors)) > 0
                    else 0
                ),
            }

        return correlations

    def generate_report(
        self, analysis_results: Dict, correlation_results: Optional[Dict] = None
    ) -> str:
        """
        Generate a human-readable report from analysis results.

        Args:
            analysis_results: Results from analyze_config_directory
            correlation_results: Optional correlation analysis results

        Returns:
            Formatted report string
        """
        report_lines = [
            "# Configuration Pattern Analysis Report",
            "",
            f"**Total configurations analyzed**: {analysis_results['total_configs']}",
            "",
        ]

        # K-fold analysis
        kfold_stats = analysis_results["kfold_analysis"]
        if kfold_stats["kfold_count"] > 0 or kfold_stats["regular_split_count"] > 0:
            kfold_rate = kfold_stats.get("kfold_rate", 0) * 100

            report_lines.extend(
                [
                    "## Data Splitting Configuration",
                    f"- **K-fold experiments**: {kfold_stats['kfold_count']} ({kfold_rate:.1f}%)",
                    f"- **Regular split experiments**: {kfold_stats['regular_split_count']} ({100-kfold_rate:.1f}%)",
                    "",
                ]
            )

            if kfold_stats.get("problematic_folds"):
                report_lines.extend(
                    [
                        f"**Frequently failing fold indices**: {', '.join(map(str, kfold_stats['problematic_folds']))}",
                        "",
                    ]
                )

        # Common parameter patterns
        if analysis_results["common_patterns"]:
            report_lines.extend(["## Common Configuration Patterns", ""])

            for param, pattern_info in analysis_results["common_patterns"].items():
                if pattern_info["most_common"] and pattern_info["diversity"] > 1:
                    most_common_val, count = pattern_info["distribution"][0]
                    percentage = (count / analysis_results["total_configs"]) * 100

                    report_lines.extend(
                        [
                            f"### {param}",
                            f"- **Most common value**: `{most_common_val}` ({count} configs, {percentage:.1f}%)",
                            f"- **Value diversity**: {pattern_info['diversity']} different values",
                            "",
                        ]
                    )

        # Error correlation analysis
        if correlation_results:
            report_lines.extend(["## Error-Configuration Correlations", ""])

            kfold_corr = correlation_results.get("kfold_error_correlation", {})
            if kfold_corr:
                kfold_rate = kfold_corr.get("kfold_error_rate", 0) * 100
                report_lines.extend(
                    [
                        f"**K-fold error rate**: {kfold_rate:.1f}% of all analyzed failures",
                        "",
                    ]
                )

                if kfold_corr.get("kfold_error_types"):
                    report_lines.extend(["**K-fold specific errors**:", ""])
                    for error_type, count in kfold_corr["kfold_error_types"][:3]:
                        report_lines.append(f"- {error_type}: {count} occurrences")
                    report_lines.append("")

        return "\n".join(report_lines)


def main():
    """Command-line interface for configuration analysis."""
    if len(sys.argv) < 2:
        print(
            "Usage: python analyze_configs.py <config_directory> [error_analysis.json]",
            file=sys.stderr,
        )
        sys.exit(1)

    config_dir = sys.argv[1]
    error_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        analyzer = ConfigurationAnalyzer()
        config_results = analyzer.analyze_config_directory(config_dir)

        correlation_results = None
        if error_file and Path(error_file).exists():
            with open(error_file) as f:
                error_analysis = json.load(f)
            correlation_results = analyzer.correlate_with_errors(
                config_results, error_analysis
            )

        # Output JSON results for programmatic use
        output = {
            "config_analysis": config_results,
            "correlation_analysis": correlation_results,
        }
        print(json.dumps(output, default=str, indent=2))

        # Also output human-readable report to stderr
        report = analyzer.generate_report(config_results, correlation_results)
        print(report, file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
