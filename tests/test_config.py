from app.config import _env


def test_env_ignores_inline_comments(monkeypatch):
    monkeypatch.setenv("CC_EMPTY", "# Nemotron Nano: claim extraction")
    monkeypatch.setenv("CC_VALUE", "nvidia/model-id   # judge")
    monkeypatch.setenv("CC_PLAIN", " plain ")
    assert _env("CC_EMPTY") == ""
    assert _env("CC_EMPTY", "fallback") == "fallback"
    assert _env("CC_VALUE") == "nvidia/model-id"
    assert _env("CC_PLAIN") == "plain"
