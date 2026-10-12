"""Held-out harness: frozen inputs, stage-level experiments, paired comparison and incremental review
(fake pipeline stages from test_run_api; no network, no models)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

import app.pipeline.run as run_mod
from app import demo
from app.models import Evidence, Passage, Reference, Verdict
from app.pipeline.run import JobRunner
from tests.test_run_api import env, fixture_ingest  # noqa: F401 - pytest fixture

_spec = importlib.util.spec_from_file_location("heldout", Path(__file__).parent.parent / "scripts" / "heldout.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def test_mcnemar_and_wilson():
    assert H.mcnemar_exact(0, 0) == 1.0
    assert H.mcnemar_exact(6, 4) == pytest.approx(0.754, abs=1e-3)
    assert H.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    lo, hi = H.wilson(89, 103)
    assert lo < 89 / 103 < hi and round(lo, 2) == 0.78 and round(hi, 2) == 0.92


@pytest.fixture
def heldout(env, tmp_path, monkeypatch):  # noqa: F811
    """A one-paper round whose baseline is a finished check of the fixture paper."""
    fakes, store = env
    monkeypatch.setattr(H, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(H, "MANIFEST", tmp_path / "eval" / "manifest.json")
    monkeypatch.setattr(H, "WORK_DIR", tmp_path / "work")
    paper = fixture_ingest()
    s1 = next(s for s in paper.sentences if s.id == "s1")
    d = tmp_path / "eval" / "p1"
    demo.write_json(d / "paper.json", paper.model_dump(mode="json"))
    truth = demo.GroundTruth(title="Fixture", planted=[demo.PlantedError(
        id="e1", sentence_id="s1", reference_keys=[s1.reference_keys[0]], mutation="scope_inflation",
        description="", original="", mutated="", expected_labels=["overstated"])])
    demo.write_json(d / "ground_truth.json", truth.model_dump(mode="json"))

    async def baseline():
        runner = JobRunner(store=store)
        job_id = await H.full_check(runner, paper, "arxiv", None)
        return runner, job_id

    runner, base_job = asyncio.run(baseline())
    demo.write_json(H.MANIFEST, {"papers": [{"slug": "p1", "kind": "arxiv", "field": "Test", "title": "Fixture",
                                             "eval_job": base_job}]})
    return fakes, store, runner, base_job


def _freeze(store, base_job, tmp_path):
    """Give one reference a cited text that contains its retrieved passage, then freeze."""
    refs = store.all(base_job, "reference", Reference)
    key = sorted(refs)[0]
    text = tmp_path / "cache" / "fulltext" / "abc.txt"
    text.parent.mkdir(parents=True)
    text.write_text("Intro. The retrieved passage. End.", encoding="utf-8")
    text.with_suffix(".refs.json").write_text('{"references": []}', encoding="utf-8")
    store.put(base_job, "reference", key, refs[key].model_copy(update={"fulltext_status": "full", "text_path": str(text)}))
    for pid, ev in store.all(base_job, "evidence", Evidence).items():
        if pid.endswith(f"|{key}"):
            store.put(base_job, "evidence", pid, ev.model_copy(update={
                "passages": [Passage(text="The retrieved passage.", chunk_index=0, score=0.9)]}))
    asyncio.run(H.cmd_freeze(Namespace(slugs=[])))
    return key


def test_freeze_copies_texts_and_checks_them_against_retrieved_passages(heldout, tmp_path, capsys):
    _, store, _, base_job = heldout
    key = _freeze(store, base_job, tmp_path)
    frozen = H.frozen_references("p1")
    assert set(frozen) == set(store.all(base_job, "reference", Reference))
    path = Path(frozen[key].text_path)
    assert path.parent == H.frozen_dir() / "texts" and path.read_text(encoding="utf-8").startswith("Intro.")
    assert path.with_suffix(".refs.json").exists()
    assert "retrieved passages found in the texts: " in capsys.readouterr().out
    # A changed text is reported.
    (tmp_path / "cache" / "fulltext" / "abc.txt").write_text("Something else.", encoding="utf-8")
    asyncio.run(H.cmd_freeze(Namespace(slugs=[])))
    assert "CHANGED" in capsys.readouterr().out


@pytest.mark.parametrize("stage, reruns", [
    ("detect", set()),
    ("judge", {"judged"}),
    ("retrieve", {"judged"}),
    ("extract", {"extracted", "judged"}),
])
def test_experiment_reruns_only_the_stages_from_its_start(heldout, tmp_path, stage, reruns):
    fakes, store, runner, base_job = heldout
    _freeze(store, base_job, tmp_path)
    fakes.extracted.clear()
    fakes.judged.clear()
    fakes.resolved.clear()

    async def go():
        job_id = await H.experiment_job(runner, H.entry(H.load_manifest(), "p1"), stage, H.BASELINE)
        runner.start(job_id)
        return job_id, await runner.wait(job_id)

    job_id, job = asyncio.run(go())
    assert job.status == "done" and fakes.resolved == []  # references always come from the frozen copy
    assert bool(fakes.extracted) == ("extracted" in reruns)
    assert bool(fakes.judged) == ("judged" in reruns)
    frozen = H.frozen_references("p1")
    assert store.all(job_id, "reference", Reference) == frozen
    assert len(store.all(job_id, "verdict", Verdict)) == len(store.all(base_job, "verdict", Verdict))


def test_experiment_compare_and_incremental_review(heldout, tmp_path, monkeypatch, capsys):
    fakes, store, runner, base_job = heldout
    _freeze(store, base_job, tmp_path)
    # The changed judge: no escalation, and s2 is now judged overstated.
    original = run_mod.judge_pair

    async def judge_pair(claim, ref, evidence, ctx):
        v = await original(claim, ref, evidence, ctx)
        return v.model_copy(update={"label": "overstated"}) if claim.citing_sentence_id == "s2" else v

    monkeypatch.setattr(run_mod, "judge_pair", judge_pair)
    monkeypatch.setattr(run_mod, "needs_escalation", lambda v, c, r: False)
    monkeypatch.setattr(H, "use_round", lambda n: None)
    monkeypatch.setattr(H, "_git_state", lambda: {"commit": "abc1234", "uncommitted_changes": False})
    asyncio.run(H.cmd_experiment(Namespace(name="no-escalation", stage="judge", base=H.BASELINE, note="test",
                                           rounds=[1], slugs=[])))
    exp = H.load_experiment("no-escalation")
    assert exp["from"] == "judge" and exp["commit"] == "abc1234" and exp["papers"]["p1"]["job_id"] != base_job
    r = exp["comparisons"]["baseline"]
    assert (r["gained"], r["lost"]) == (1, 0)
    assert r["planted"][0]["id"] == "e1" and r["totals"]["run"]["recall"] == 1.0
    s2_keys = {k for s in fixture_ingest().sentences if s.id == "s2" for k in s.reference_keys}
    assert {(x["sentence_id"], x["reference_key"], x["change"], x["review"]) for x in r["flags"]} == {
        ("s2", k, "added", "unreviewed") for k in s2_keys}
    assert "gained 1, lost 0" in capsys.readouterr().out

    # Re-running the same experiment resumes it instead of starting new jobs.
    fakes.judged.clear()
    asyncio.run(H.cmd_experiment(Namespace(name="no-escalation", stage="judge", base=H.BASELINE, note=None,
                                           rounds=[1], slugs=[])))
    assert fakes.judged == [] and H.load_experiment("no-escalation")["papers"]["p1"]["job_id"] == exp["papers"]["p1"]["job_id"]

    # Incremental review: only the new flags, appended to the annotations without losing earlier ones.
    ann_path = H.paper_dir("p1") / "annotations.json"
    demo.write_json(ann_path, {"paper": "p1", "annotations": [
        {"sentence_id": "s3", "reference_key": "x", "verdict": "false_alarm", "note": "earlier review"}]})
    H.cmd_review_new(Namespace(name="no-escalation", slugs=[]))
    sheet = (H.paper_dir("p1") / "review-no-escalation.md").read_text(encoding="utf-8")
    assert sheet.count("\n## s2 ") == len(s2_keys) and "## s1 " not in sheet  # s1 is planted
    entries = json.loads(ann_path.read_text(encoding="utf-8"))["annotations"]
    assert entries[-1]["sentence_id"] == "s3" and entries[-1]["verdict"] == "false_alarm"
    assert {(e["sentence_id"], e["verdict"]) for e in entries[:-1]} == {("s2", "")}

    # Once reviewed, the flags count in precision and no sheet is written.
    for e in entries:
        if e["sentence_id"] == "s2":
            e.update(verdict="genuine", note="ok")
    demo.write_json(ann_path, {"paper": "p1", "annotations": entries})
    H.cmd_review_new(Namespace(name="no-escalation", slugs=[]))
    assert not (H.paper_dir("p1") / "review-no-escalation.md").exists()
    r = H.compare_round(JobRunner(store=store), "no-escalation", H.BASELINE)
    assert r["totals"]["run"]["genuine"] == len(s2_keys) and r["totals"]["run"]["unreviewed"] == 0
    assert r["totals"]["run"]["precision_reviewed"] == 1.0

    # The experiment becomes the development baseline; the evaluation check itself is kept.
    H.cmd_set_baseline(Namespace(name="no-escalation"))
    p = H.entry(H.load_manifest(), "p1")
    assert H.run_job_id(p, H.BASELINE) == exp["papers"]["p1"]["job_id"] and p["eval_job"] == base_job
    assert H.load_manifest()["dev_baseline"]["run"] == "no-escalation"
    assert exp["papers"]["p1"]["base_job"] == base_job


def test_experiment_names_are_checked(heldout):
    with pytest.raises(SystemExit):
        asyncio.run(H.cmd_experiment(Namespace(name="baseline", stage="judge", base=H.BASELINE, note=None,
                                               rounds=[1], slugs=[])))
    with pytest.raises(SystemExit):
        asyncio.run(H.cmd_experiment(Namespace(name="../x", stage="judge", base=H.BASELINE, note=None,
                                               rounds=[1], slugs=[])))

