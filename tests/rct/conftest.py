import pytest
import ray


@pytest.fixture(scope="session", autouse=True)
def ray_cluster():
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()
