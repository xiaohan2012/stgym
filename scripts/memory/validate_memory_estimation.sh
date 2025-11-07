#!/bin/bash
# Wrapper script to validate memory estimation accuracy from any directory
# Example:
# ./scripts/memory/validate_memory_estimation.sh --config conf/adhoc/test.yaml --cpu-only

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Set PYTHONPATH and run the script from project root
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT" python scripts/memory/validate_memory_estimation.py "$@"
