"""Tests for DatasetLoadGate actor and gated_datasets integration in run_exp."""

from unittest.mock import Mock, patch

import pytest
import ray

from stgym.config_schema import (
    DataLoaderConfig,
    ExperimentConfig,
    GraphClassifierModelConfig,
    LRScheduleConfig,
    MessagePassingConfig,
    MLFlowConfig,
    OptimizerConfig,
    PoolingConfig,
    PostMPConfig,
    TaskConfig,
    TrainConfig,
)
from stgym.rct.run import run_exp
from stgym.utils import DatasetLoadGate, gated_load, rm_dir_if_exists

# ---------------------------------------------------------------------------
# Gate actor tests (require a live Ray cluster)
# ---------------------------------------------------------------------------


class TestDatasetLoadGate:
    def test_single_acquire_release(self):
        """Basic acquire/release round-trip completes without error."""
        gate = DatasetLoadGate.remote(max_concurrent=1)
        ray.get(gate.acquire.remote())
        gate.release.remote()

    def test_second_acquire_blocked_until_release(self):
        """With max_concurrent=1, a second acquire only completes after release."""
        gate = DatasetLoadGate.remote(max_concurrent=1)

        # First acquire — should succeed immediately.
        ray.get(gate.acquire.remote())

        # Second acquire — submit but do NOT block yet.
        second = gate.acquire.remote()

        # Release the semaphore; now second should unblock.
        gate.release.remote()

        # This should now complete quickly (would hang if gate stayed locked).
        ray.get(second, timeout=5)

    def test_named_gate_get_if_exists(self):
        """get_if_exists=True returns the same actor, not a new one."""
        gate1 = DatasetLoadGate.options(name="test_gate_singleton").remote(
            max_concurrent=1
        )
        gate2 = DatasetLoadGate.options(
            name="test_gate_singleton", get_if_exists=True
        ).remote()
        # Both handles should refer to the same actor — acquiring on one
        # should be visible to the other.
        ray.get(gate1.acquire.remote())
        second = gate2.acquire.remote()
        gate1.release.remote()
        ray.get(second, timeout=5)


# ---------------------------------------------------------------------------
# run_exp gate integration tests (no Ray, mocked gate)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def teardown():
    yield
    rm_dir_if_exists("tests/data/brca-test/processed")


@patch("stgym.rct.run.train")
@patch("stgym.rct.run.STGymModule")
@patch("stgym.rct.run.STDataModule")
@patch("stgym.rct.run.gated_load")
class TestRunExpGating:
    """Verify that run_exp calls gated_load with the correct arguments."""

    GATED = frozenset(["mouse-kidney"])

    @staticmethod
    def _make_exp_cfg(dataset_name: str, num_classes: int) -> ExperimentConfig:
        task = TaskConfig(
            dataset_name=dataset_name,
            type="graph-classification",
            num_classes=num_classes,
        )
        dl_cfg = DataLoaderConfig(batch_size=4)
        dl_cfg.split = DataLoaderConfig.DataSplitConfig(
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        model = GraphClassifierModelConfig(
            mp_layers=[
                MessagePassingConfig(
                    layer_type="gcnconv",
                    pooling=PoolingConfig(type="dmon", n_clusters=8),
                )
            ],
            global_pooling="mean",
            post_mp_layer=PostMPConfig(dims=[16, 8]),
        )
        train_cfg = TrainConfig(
            optim=OptimizerConfig(),
            lr_schedule=LRScheduleConfig(type=None),
            max_epoch=1,
            early_stopping={"metric": "val_roc_auc", "mode": "max"},
        )
        return ExperimentConfig(
            task=task, data_loader=dl_cfg, model=model, train=train_cfg
        )

    @pytest.mark.parametrize(
        "dataset_name,num_classes",
        [
            ("mouse-kidney", 3),
            ("brca-test", 2),
        ],
    )
    def test_gated_load_called_with_dataset_name(
        self,
        mock_gated_load,
        mock_data_module,
        mock_model,
        mock_train,
        dataset_name,
        num_classes,
    ):
        exp_cfg = self._make_exp_cfg(dataset_name, num_classes)
        run_exp(exp_cfg, MLFlowConfig(track=False), gated_datasets=self.GATED)
        mock_gated_load.assert_called_once_with(dataset_name, self.GATED)


class TestGatedLoad:
    """Verify gated_load context manager acquires/releases the gate correctly."""

    GATED = frozenset(["mouse-kidney"])

    def test_noop_for_ungated_dataset(self):
        """Ungated dataset passes through without touching any actor."""
        with gated_load("brca-test", self.GATED):
            pass  # no error, no actor needed

    @patch("stgym.utils.ray")
    @patch("stgym.utils.DatasetLoadGate")
    def test_releases_even_on_exception(self, mock_gate_cls, mock_ray):
        """Gate is released via finally even when the body raises."""
        mock_gate = Mock()
        mock_gate_cls.options.return_value.remote.return_value = mock_gate
        mock_ray.get = Mock()

        with pytest.raises(RuntimeError):
            with gated_load("mouse-kidney", self.GATED):
                raise RuntimeError("load failed")

        mock_gate.release.remote.assert_called_once()
