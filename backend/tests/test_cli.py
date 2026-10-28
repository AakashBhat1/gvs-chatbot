from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from app import check_tree, cli

WriteTree = Callable[[dict[str, Any]], Path]
TODAY = date(2026, 10, 29)


# ----------------------------------------------------------------- gvs-resolve


def test_resolve_text_output(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "--country",
            "japan",
            "--purpose",
            "tourism",
            "-t",
            "relation=self,profession=employed",
            "-t",
            "relation=son,age=3",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "== Japan - Tourism" in out
    assert "Shared, once per application:" in out
    assert "2. Son (School student, age 3)" in out
    assert "WARNING: document data status is 'placeholder'" in out


def test_resolve_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "--country",
            "us",
            "--purpose",
            "business",
            "-t",
            "relation=self,profession=freelancer",
            "--json",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    docs = [d["id"] for d in data["groups"][0]["travellers"][0]["documents"]]
    assert "ds160_confirmation" in docs
    assert "business_invitation" in docs


def test_resolve_reports_missing_profession(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["--country", "japan", "--purpose", "tourism", "-t", "relation=mother,age=60"])
    assert code == 2
    assert "missing profession[0]" in capsys.readouterr().err


def test_resolve_rejects_unknown_ids(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        ["--country", "mars", "--purpose", "tourism", "-t", "relation=self,profession=employed"]
    )
    assert code == 2
    assert "unknown id 'mars'" in capsys.readouterr().err


def test_resolve_bad_tree(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "t.json"
    bad.write_text("{}", encoding="utf-8")
    code = cli.main(
        ["--tree", str(bad), "--country", "jp", "--purpose", "x", "-t", "relation=self"]
    )
    assert code == 1
    assert "visa-tree.json is invalid" in capsys.readouterr().err


def test_resolve_caps_traveller_count(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--country", "japan", "--purpose", "tourism"]
    for _ in range(9):
        args += ["-t", "relation=self,profession=employed"]
    assert cli.main(args) == 2
    assert "at most 8" in capsys.readouterr().err


@pytest.mark.parametrize(
    "spec",
    [
        "relation",
        "colour=red",
        "relation=self,relation=son",
        "relation=self,age=three",
        "relation=Self!",
        "relation=",
    ],
)
def test_bad_traveller_spec_exits(spec: str) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--country", "japan", "--purpose", "tourism", "-t", spec])
    assert exc.value.code == 2


def test_parse_traveller_full() -> None:
    t = cli.parse_traveller("relation=mother,profession=retired,age=61,country=uk,purpose=business")
    assert (t.relation, t.profession, t.age, t.country, t.purpose) == (
        "mother",
        "retired",
        61,
        "uk",
        "business",
    )


# ----------------------------------------------------------------- gvs-check-tree


def test_check_real_tree_passes_dev_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert check_tree.main([], today=TODAY) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("OK")
    assert "15 countries, 3 purposes, 17 professions, 10 relations" in captured.out
    assert "never been reviewed" in captured.err  # warned, not failed


def test_check_real_tree_fails_release_gate(capsys: pytest.CaptureFixture[str]) -> None:
    assert check_tree.main(["--release"], today=TODAY) == 1
    err = capsys.readouterr().err
    assert "meta.status is 'placeholder'" in err
    assert "[PLACEHOLDER]" in err


def test_check_release_passes_for_reviewed_tree(
    mini_dict: dict[str, Any], write_tree: WriteTree, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_tree(mini_dict)
    assert check_tree.main([str(path), "--release"], today=TODAY) == 0
    assert capsys.readouterr().out.startswith("OK")


@pytest.mark.parametrize(
    ("reviewed", "message"),
    [("2026-01-01", "days ago"), ("2027-01-01", "in the future"), (None, "never been reviewed")],
)
def test_check_release_review_dates(
    mini_dict: dict[str, Any],
    write_tree: WriteTree,
    capsys: pytest.CaptureFixture[str],
    reviewed: str | None,
    message: str,
) -> None:
    mini_dict["countries"]["uk"]["last_reviewed"] = reviewed
    path = write_tree(mini_dict)
    assert check_tree.main([str(path), "--release"], today=TODAY) == 1
    assert message in capsys.readouterr().err


def test_check_warns_unused_document(
    mini_dict: dict[str, Any], write_tree: WriteTree, capsys: pytest.CaptureFixture[str]
) -> None:
    mini_dict["documents"]["orphan"] = {"label": "Orphan"}
    assert check_tree.main([str(write_tree(mini_dict))], today=TODAY) == 0
    assert "'orphan' is never used" in capsys.readouterr().err


def test_check_invalid_tree_fails(
    mini_dict: dict[str, Any], write_tree: WriteTree, capsys: pytest.CaptureFixture[str]
) -> None:
    mini_dict["rules"]["mars"] = ["nope"]
    assert check_tree.main([str(write_tree(mini_dict))], today=TODAY) == 1
    err = capsys.readouterr().err
    assert "unknown country 'mars'" in err
    assert "unknown document 'nope'" in err


def test_default_tree_path_env_override(
    monkeypatch: pytest.MonkeyPatch, mini_dict: dict[str, Any], write_tree: WriteTree
) -> None:
    tree_path = write_tree(mini_dict)
    lexicon = tree_path.with_name("lexicon.toml")
    lexicon.write_text('[countries]\njp = ["japan"]\n', encoding="utf-8")
    monkeypatch.setenv("GVS_TREE_PATH", str(tree_path))
    monkeypatch.setenv("GVS_LEXICON_PATH", str(lexicon))
    monkeypatch.setenv("GVS_GENERATED_LEXICON_PATH", str(tree_path.with_name("none.json")))
    assert check_tree.main(["--release"], today=TODAY) == 0


def test_check_fails_on_bad_lexicon(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "lexicon.toml"
    bad.write_text('[countries]\natlantis = ["atlantis"]\n', encoding="utf-8")
    assert check_tree.main(["--lexicon", str(bad)], today=TODAY) == 1
    assert "lexicon: countries.atlantis: not an id" in capsys.readouterr().err
