from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import trial

CASES = [
    {
        "id": "ok",
        "message": "just me, holiday in Vienna, I am a software engineer",
        "expect": {
            "form": {
                "country": "austria",
                "purpose": "tourism",
                "travellers": [{"relation": "self", "profession": "employed"}],
            }
        },
    },
    {"id": "miss", "message": "Japan", "expect": {"form": {"country": "uk"}}},
]


@pytest.fixture
def case_set(tmp_path: Path) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in CASES), encoding="utf-8")
    return path


def test_trial_saves_replies_checklists_and_comparison(case_set: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    args = ["--brains", "rules", "--set", str(case_set), "--out", str(out)]
    assert trial.main([*args, "--label", "a"]) == 0
    assert trial.main([*args, "--label", "b"]) == 0

    ok, miss = json.loads((out / "a.json").read_text(encoding="utf-8"))["rows"]
    assert ok["exact"] is True
    assert ok["brain"] == "rules"
    assert ok["reply"].startswith("Here's the summary:")
    assert ok["checklist"][0]["country"] == "austria"
    assert ok["checklist"][0]["placeholders"] == ["application_form"]
    assert miss["exact"] is False
    assert miss["wrong"] == ["country"]
    assert miss["checklist"] == []

    assert trial.main(["--compare", str(out)]) == 0
    report = (out / "comparison.md").read_text(encoding="utf-8")
    assert "| run | country |" in report
    assert "| miss | Japan | country | country |" in report
    assert "### miss [MISS] Japan" in report
    assert "bot said: Here's the summary:" in report


def test_compare_needs_runs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no runs"):
        trial.compare(tmp_path)
