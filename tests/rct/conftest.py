import pytest


@pytest.fixture(scope="session", autouse=True)
def _init_ray(ray_cluster):
    """Auto-initialize Ray for all tests in this directory."""
