from pathlib import Path

from app.config import ROOT_DIR, _env, portable_path, project_path


def test_env_ignores_inline_comments(monkeypatch):
    monkeypatch.setenv("CC_EMPTY", "# Nemotron Nano: claim extraction")
    monkeypatch.setenv("CC_VALUE", "nvidia/model-id   # judge")
    monkeypatch.setenv("CC_PLAIN", " plain ")
    assert _env("CC_EMPTY") == ""
    assert _env("CC_EMPTY", "fallback") == "fallback"
    assert _env("CC_VALUE") == "nvidia/model-id"
    assert _env("CC_PLAIN") == "plain"


def test_portable_paths_carry_no_local_directory(tmp_path):
    inside = ROOT_DIR / ".cache" / "fulltext" / "abc.txt"
    assert portable_path(str(inside)) == ".cache/fulltext/abc.txt"
    assert portable_path(str(tmp_path / "x" / "y.txt")) == "y.txt"  # outside the project: file name only
    assert portable_path(None) is None
    assert project_path(".cache/fulltext/abc.txt") == inside
    assert project_path(str(inside)) == inside


def test_reference_text_read_from_a_project_relative_path(monkeypatch, tmp_path):
    import app.pipeline.references as refs
    from app.models import Reference

    (tmp_path / "t.txt").write_text("cited text", encoding="utf-8")
    (tmp_path / "t.refs.json").write_text('{"references": []}', encoding="utf-8")
    monkeypatch.setattr(refs, "project_path", lambda p: tmp_path / Path(p).name)
    ref = Reference(key="r", raw="", text_path=".cache/fulltext/t.txt")
    assert refs.read_reference_text(ref) == "cited text"
    assert refs.read_reference_sidecar(ref) == {"references": []}
