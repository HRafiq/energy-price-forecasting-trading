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


@pytest.fixture(autouse=True)
def _no_real_model_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """No test reads the real .env or reaches a paid model.

    The briefing endpoint looks for keys in the environment and in the repository's
    .env. Once a real key is pasted there, a test run would otherwise make billed
    calls and fail on prose that differs every time.
    """
    from src.narration import provider

    for name in (
        provider.API_KEY_ENV,
        provider.ANTHROPIC_KEY_ENV,
        provider.PROVIDER_ENV,
        provider.MODEL_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        provider, "REPO_ROOT", tmp_path_factory.getbasetemp() / "no_keys"
    )
