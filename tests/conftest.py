from __future__ import annotations

import pytest

from src.config import Settings, SmardConfig, load_settings


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def smard_config(settings: Settings) -> SmardConfig:
    return settings.smard.model_copy(
        update={"max_workers": 1, "refresh_recent_chunks": 1}
    )
