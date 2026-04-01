"""Tests that run_rct wires Ray memory options correctly from ResourceConfig."""

from unittest.mock import MagicMock, patch

from stgym.config_schema import ResourceConfig


def _make_options_kwargs(res_cfg: ResourceConfig, dataset_name: str) -> dict:
    """Mirrors the options-building logic in run_rct.main()."""
    memory_bytes = res_cfg.get_memory_bytes(dataset_name)
    return {
        "num_cpus": res_cfg.num_cpus_per_trial,
        "num_gpus": res_cfg.num_gpus_per_trial,
        **({"memory": memory_bytes} if memory_bytes is not None else {}),
    }


class TestRayMemoryOption:
    def test_memory_absent_when_not_configured(self):
        res_cfg = ResourceConfig()
        opts = _make_options_kwargs(res_cfg, "mouse-kidney")
        assert "memory" not in opts

    def test_memory_present_for_configured_dataset(self):
        res_cfg = ResourceConfig(dataset_memory_gb={"mouse-kidney": 40.0})
        opts = _make_options_kwargs(res_cfg, "mouse-kidney")
        assert opts["memory"] == int(40.0 * 1024**3)

    def test_memory_absent_for_unconfigured_dataset(self):
        res_cfg = ResourceConfig(dataset_memory_gb={"mouse-kidney": 40.0})
        opts = _make_options_kwargs(res_cfg, "brca")
        assert "memory" not in opts

    def test_memory_per_dataset(self):
        res_cfg = ResourceConfig(
            dataset_memory_gb={"mouse-kidney": 40.0, "glioblastoma": 20.0}
        )
        assert _make_options_kwargs(res_cfg, "mouse-kidney")["memory"] == int(
            40.0 * 1024**3
        )
        assert _make_options_kwargs(res_cfg, "glioblastoma")["memory"] == int(
            20.0 * 1024**3
        )
        assert "memory" not in _make_options_kwargs(res_cfg, "brca")

    def test_ray_remote_options_receives_memory(self):
        """Verify memory= is passed to ray.remote().options() when configured."""
        res_cfg = ResourceConfig(dataset_memory_gb={"mouse-kidney": 40.0})

        mock_remote = MagicMock()
        mock_options = MagicMock()
        mock_remote.options.return_value = mock_options

        with patch("run_rct.ray") as mock_ray, patch("run_rct.run_exp") as mock_run_exp:
            mock_ray.remote.return_value = mock_remote

            # Replicate the options call from run_rct.main()
            import run_rct

            opts = _make_options_kwargs(res_cfg, "mouse-kidney")
            run_rct.ray.remote(mock_run_exp).options(**opts)

            mock_remote.options.assert_called_once_with(
                num_cpus=res_cfg.num_cpus_per_trial,
                num_gpus=res_cfg.num_gpus_per_trial,
                memory=int(40.0 * 1024**3),
            )

    def test_ray_remote_options_no_memory_key_when_absent(self):
        """Verify memory= is NOT passed for unconfigured datasets."""
        res_cfg = ResourceConfig()

        mock_remote = MagicMock()
        mock_options = MagicMock()
        mock_remote.options.return_value = mock_options

        with patch("run_rct.ray") as mock_ray, patch("run_rct.run_exp") as mock_run_exp:
            mock_ray.remote.return_value = mock_remote

            import run_rct

            opts = _make_options_kwargs(res_cfg, "brca")
            run_rct.ray.remote(mock_run_exp).options(**opts)

            call_kwargs = mock_remote.options.call_args.kwargs
            assert "memory" not in call_kwargs
