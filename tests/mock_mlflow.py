"""Shared mock classes for MLflow testing."""

from dataclasses import dataclass, field


@dataclass
class MockRunData:
    """Mock MLflow run data structure for testing."""

    tags: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)


@dataclass
class MockRunInfo:
    """Mock MLflow run info structure for testing."""

    status: str = "FINISHED"


@dataclass
class MockRun:
    """Mock MLflow Run object for testing."""

    data: MockRunData = field(default_factory=MockRunData)
    info: MockRunInfo = field(default_factory=MockRunInfo)
