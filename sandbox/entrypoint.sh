#!/bin/bash
set -e

echo "Installing dependencies..."
uv sync --group dev --reinstall
echo "Installing pre-commit hooks..."
pre-commit install --install-hooks
echo "Ready."

exec bash
