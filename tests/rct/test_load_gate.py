"""Tests for DatasetLoadGate actor and gated_datasets integration in run_exp."""

import asyncio
from unittest.mock import Mock, call, patch

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
from stgym.utils import DatasetLoadGate, rm_dir_if_exists


# ---------------------------------------------------------------------------
# Gate actor tests (require a live Ray cluster)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def ray_cluster():
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()


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


@pytest.fixture
def exp_cfg_gated():
    """ExperimentConfig whose dataset_name is in gated_datasets."""
    task = TaskConfig(
        dataset_name="mouse-kidney", type="graph-classification", num_classes=3
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
    return ExperimentConfig(task=task, data_loader=dl_cfg, model=model, train=train_cfg)


@pytest.fixture
def exp_cfg_ungated():
    """ExperimentConfig whose dataset_name is NOT in gated_datasets."""
    task = TaskConfig(
        dataset_name="brca-test", type="graph-classification", num_classes=2
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
    return ExperimentConfig(task=task, data_loader=dl_cfg, model=model, train=train_cfg)


@pytest.fixture(autouse=True)
def teardown():
    yield
    rm_dir_if_exists("tests/data/brca-test/processed")


class TestRunExpGating:
    """Verify that run_exp acquires/releases the gate iff dataset is gated."""

    GATED = frozenset(["mouse-kidney"])

    def _make_mock_gate(self):
        mock_gate = Mock()
        mock_acquire_ref = Mock()
        mock_gate.acquire.return_value = mock_acquire_ref
        return mock_gate

    @patch("stgym.rct.run.train")
    @patch("stgym.rct.run.STGymModule")
    @patch("stgym.rct.run.STDataModule")
    @patch("stgym.rct.run.ray")
    @patch("stgym.rct.run.DatasetLoadGate")
    def test_gate_acquired_for_gated_dataset(
        self,
        mock_gate_cls,
        mock_ray,
        mock_data_module,
        mock_model,
        mock_train,
        exp_cfg_gated,
    ):
        mock_gate = self._make_mock_gate()
        mock_gate_cls.options.return_value.remote.return_value = mock_gate
        mock_ray.get = Mock()

        run_exp(
            exp_cfg_gated,
            MLFlowConfig(track=False),
            gated_datasets=self.GATED,
        )

        mock_gate_cls.options.assert_called_once_with(
            name="dataset_load_gate", get_if_exists=True
        )
        # Gate uses Ray remote pattern: acquire.remote() and release.remote()
        mock_gate.acquire.remote.assert_called_once()
        mock_gate.release.remote.assert_called_once()

    @patch("stgym.rct.run.train")
    @patch("stgym.rct.run.STGymModule")
    @patch("stgym.rct.run.STDataModule")
    @patch("stgym.rct.run.ray")
    @patch("stgym.rct.run.DatasetLoadGate")
    def test_gate_not_used_for_ungated_dataset(
        self,
        mock_gate_cls,
        mock_ray,
        mock_data_module,
        mock_model,
        mock_train,
        exp_cfg_ungated,
    ):
        run_exp(
            exp_cfg_ungated,
            MLFlowConfig(track=False),
            gated_datasets=self.GATED,
        )

        mock_gate_cls.options.assert_not_called()

    @patch("stgym.rct.run.train")
    @patch("stgym.rct.run.STGymModule")
    @patch("stgym.rct.run.STDataModule")
    @patch("stgym.rct.run.ray")
    @patch("stgym.rct.run.DatasetLoadGate")
    def test_gate_released_even_if_data_load_raises(
        self,
        mock_gate_cls,
        mock_ray,
        mock_data_module,
        mock_model,
        mock_train,
        exp_cfg_gated,
    ):
        mock_gate = self._make_mock_gate()
        mock_gate_cls.options.return_value.remote.return_value = mock_gate
        mock_ray.get = Mock()
        mock_data_module.side_effect = RuntimeError("load failed")

        run_exp(
            exp_cfg_gated,
            MLFlowConfig(track=False),
            gated_datasets=self.GATED,
        )

        # Gate must be released even when data loading raises.
        mock_gate.release.remote.assert_called_once()
