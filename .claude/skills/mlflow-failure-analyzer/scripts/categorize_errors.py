#!/usr/bin/env python3
"""
Error categorization script for MLflow experiment failures.
Analyzes error logs to identify patterns and classify failure types.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict


class ErrorCategorizer:
    """Categorizes MLflow experiment errors into common failure types."""

    def __init__(self):
        """Initialize error patterns based on real MLflow failure analysis."""
        self.error_patterns = {
            "cuda_nvml": {
                "patterns": [
                    r"NVML_SUCCESS == DriverAPI::get\(\)->nvmlDeviceGetHandleByPciBusId_v2_",
                    r"INTERNAL ASSERT FAILED.*CUDACachingAllocator\.cpp",
                    r"CUDA.*out of memory",
                    r"nvmlDeviceGetHandleByPciBusId",
                ],
                "description": "CUDA/GPU driver and memory allocation issues",
                "severity": "high",
            },
            "early_stopping_validation": {
                "patterns": [
                    r"Early stopping conditioned on metric.*which is not available",
                    r"val_loss.*which is not available",
                    r"EarlyStopping.*metric.*not available",
                ],
                "description": "Missing validation metrics for early stopping",
                "severity": "high",
            },
            "kfold_validation": {
                "patterns": [
                    r"Training failed in fold \d+",
                    r"split_\d+.*which is not available",
                    r"k.*fold.*validation.*error",
                ],
                "description": "K-fold cross-validation configuration issues",
                "severity": "medium",
            },
            "nan_loss": {
                "patterns": [
                    r"loss.*nan",
                    r"NaN.*loss",
                    r"gradient.*nan",
                    r"inf.*loss",
                ],
                "description": "Training loss became NaN or infinite",
                "severity": "medium",
            },
            "memory_error": {
                "patterns": [
                    r"RuntimeError.*memory",
                    r"OutOfMemoryError",
                    r"MemoryError",
                    r"out of memory",
                ],
                "description": "System memory exhaustion",
                "severity": "high",
            },
            "timeout_error": {
                "patterns": [
                    r"TimeoutError",
                    r"timeout.*exceeded",
                    r"connection.*timeout",
                ],
                "description": "Operation timeouts",
                "severity": "medium",
            },
            "dimension_mismatch": {
                "patterns": [
                    r"dimension.*mismatch",
                    r"size mismatch",
                    r"shape.*incompatible",
                    r"RuntimeError.*size",
                ],
                "description": "Tensor dimension or shape incompatibilities",
                "severity": "medium",
            },
            "index_error": {
                "patterns": [
                    r"IndexError",
                    r"index.*out of.*range",
                    r"list index out of range",
                ],
                "description": "Array or list index out of bounds",
                "severity": "medium",
            },
            "configuration_error": {
                "patterns": [
                    r"ConfigurationError",
                    r"Invalid.*configuration",
                    r"KeyError.*config",
                    r"missing.*required.*parameter",
                ],
                "description": "Invalid or missing configuration parameters",
                "severity": "high",
            },
        }

    def categorize_error(self, error_text: str) -> Dict:
        """
        Categorize a single error text into failure types.

        Args:
            error_text: Raw error log content

        Returns:
            Dict with error category, confidence, and matched patterns
        """
        if not error_text:
            return {
                "category": "unknown",
                "confidence": 0.0,
                "matched_patterns": [],
                "description": "No error text provided",
            }

        # Convert to lowercase for case-insensitive matching
        text_lower = error_text.lower()

        # Track matches for each category
        category_scores = defaultdict(list)

        for category, config in self.error_patterns.items():
            for pattern in config["patterns"]:
                matches = re.findall(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                if matches:
                    category_scores[category].append(
                        {
                            "pattern": pattern,
                            "matches": len(matches),
                            "severity": config["severity"],
                        }
                    )

        if not category_scores:
            return {
                "category": "uncategorized",
                "confidence": 0.0,
                "matched_patterns": [],
                "description": "No known error patterns detected",
            }

        # Calculate confidence scores
        best_category = None
        best_score = 0

        for category, matches in category_scores.items():
            # Score based on number of patterns matched and severity
            score = len(matches)
            if any(m["severity"] == "high" for m in matches):
                score *= 1.5

            if score > best_score:
                best_score = score
                best_category = category

        return {
            "category": best_category,
            "confidence": min(best_score / 3.0, 1.0),  # Normalize to 0-1
            "matched_patterns": [m["pattern"] for m in category_scores[best_category]],
            "description": self.error_patterns[best_category]["description"],
            "severity": self.error_patterns[best_category]["severity"],
        }

    def analyze_error_directory(self, error_dir: str) -> Dict:
        """
        Analyze all error files in a directory.

        Args:
            error_dir: Path to directory containing error log files

        Returns:
            Dict with aggregated error analysis results
        """
        error_path = Path(error_dir)
        if not error_path.exists():
            raise FileNotFoundError(f"Error directory not found: {error_dir}")

        results = {
            "total_errors": 0,
            "categorized_errors": defaultdict(list),
            "category_counts": Counter(),
            "uncategorized_count": 0,
            "summary": {},
            "file_analysis": {},
        }

        # Process each error file
        error_files = list(error_path.glob("*.txt"))

        for error_file in error_files:
            try:
                with open(error_file, encoding="utf-8") as f:
                    error_text = f.read()

                # Categorize this error
                categorization = self.categorize_error(error_text)

                # Store results
                results["file_analysis"][error_file.name] = categorization
                results["total_errors"] += 1

                if categorization["category"] in ["unknown", "uncategorized"]:
                    results["uncategorized_count"] += 1
                else:
                    results["categorized_errors"][categorization["category"]].append(
                        {
                            "file": error_file.name,
                            "confidence": categorization["confidence"],
                            "description": categorization["description"],
                        }
                    )
                    results["category_counts"][categorization["category"]] += 1

            except Exception as e:
                print(f"Error processing {error_file}: {e}", file=sys.stderr)

        # Generate summary statistics
        if results["total_errors"] > 0:
            categorized_rate = (
                results["total_errors"] - results["uncategorized_count"]
            ) / results["total_errors"]
            results["summary"] = {
                "categorization_rate": categorized_rate,
                "most_common_error": (
                    results["category_counts"].most_common(1)[0]
                    if results["category_counts"]
                    else None
                ),
                "error_diversity": len(results["category_counts"]),
                "avg_confidence": sum(
                    analysis["confidence"]
                    for analysis in results["file_analysis"].values()
                )
                / results["total_errors"],
            }

        return results

    def generate_report(self, analysis_results: Dict) -> str:
        """
        Generate a human-readable report from analysis results.

        Args:
            analysis_results: Results from analyze_error_directory

        Returns:
            Formatted report string
        """
        report_lines = [
            "# Error Categorization Analysis Report",
            "",
            f"**Total errors analyzed**: {analysis_results['total_errors']}",
            f"**Successfully categorized**: {analysis_results['total_errors'] - analysis_results['uncategorized_count']}",
            f"**Categorization rate**: {analysis_results['summary'].get('avg_confidence', 0):.2%}",
            "",
        ]

        if analysis_results["category_counts"]:
            report_lines.extend(["## Error Type Breakdown", ""])

            # Sort by frequency
            for category, count in analysis_results["category_counts"].most_common():
                percentage = (count / analysis_results["total_errors"]) * 100
                category_info = analysis_results["categorized_errors"][category][
                    0
                ]  # Get first example

                report_lines.extend(
                    [
                        f"### {category.replace('_', ' ').title()} ({count} errors, {percentage:.1f}%)",
                        f"- **Description**: {category_info['description']}",
                        f"- **Severity**: {self.error_patterns[category]['severity']}",
                        f"- **Representative patterns**: {', '.join(self.error_patterns[category]['patterns'][:2])}",
                        "",
                    ]
                )

        if analysis_results["uncategorized_count"] > 0:
            report_lines.extend(
                [
                    f"## Uncategorized Errors ({analysis_results['uncategorized_count']} errors)",
                    "These errors did not match known patterns and may require manual investigation.",
                    "",
                ]
            )

        return "\n".join(report_lines)


def main():
    """Command-line interface for error categorization."""
    if len(sys.argv) != 2:
        print("Usage: python categorize_errors.py <error_directory>", file=sys.stderr)
        sys.exit(1)

    error_dir = sys.argv[1]

    try:
        categorizer = ErrorCategorizer()
        results = categorizer.analyze_error_directory(error_dir)

        # Output JSON results for programmatic use
        print(json.dumps(results, default=str, indent=2))

        # Also output human-readable report to stderr
        report = categorizer.generate_report(results)
        print(report, file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
