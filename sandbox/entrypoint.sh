#!/bin/bash
set -e

# Trust the mounted repo directory (owned by host user, not container user)
git config --global --add safe.directory /repo

echo "Installing dependencies..."
uv sync --group dev --reinstall
echo "Installing pre-commit hooks..."
pre-commit install --install-hooks
echo "Setting up Claude Code plugins..."
bash /repo/scripts/setup-sandbox-plugins.sh || echo "Plugin setup failed — run scripts/setup-sandbox-plugins.sh manually to retry."
echo "Ready."

exec bash
