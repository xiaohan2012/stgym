"""Tests for the Marimo notebook rct_experiment_analysis.py."""

import importlib.util
from pathlib import Path
from unittest import mock

NOTEBOOK_PATH = Path(__file__).parent.parent / "rct_experiment_analysis.py"


def _load_notebook_module():
    """Import the notebook as a Python module and return it."""
    spec = importlib.util.spec_from_file_location(
        "rct_experiment_analysis", NOTEBOOK_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMarimoNotebook:
    """Tests for the Marimo notebook via programmatic execution."""

    def test_notebook_importable(self):
        """Verify the notebook can be imported and exposes a marimo App object."""
        module = _load_notebook_module()

        assert hasattr(module, "app")
        assert hasattr(module.app, "run")

    @mock.patch("stgym.rct_utils.fetch_runs")
    def test_notebook_runs_with_no_data(self, mock_fetch_runs):
        """Verify notebook runs without error when no MLflow data is available.

        With an empty experiment_id (the default), mo.stop() halts execution
        early before any MLflow calls are made.
        """
        mock_fetch_runs.return_value = []
        module = _load_notebook_module()

        result = module.app.run()

        assert result is not None
