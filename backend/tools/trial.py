"""Run a message set through the whole bot and save what came out, message by message.

``gvs-eval`` gives scores; this also shows the reply a visitor would get and, when a
message completes the form, the checklist behind it. One file per run, then a
side-by-side comparison::

    uv run python -m tools.trial --brains rules --label rules --out out/
    uv run python -m tools.trial --brains local --solo --label model --out out/
    uv run python -m tools.trial --compare out/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.brains import Brain, BrainResult
from app.brains.factory import brain_names, build_cascade
from app.brains.lexicon import load_lexicon
from app.chat import BotReply, ChatEngine, ChatState
from app.evaluate import DEFAULT_SET, FIELDS, Case, load_cases, run_case, score
from app.extraction import Context
from app.paths import default_lexicon_path, default_tree_path
from app.resolver import Checklist
from app.tree import LoadedTree, load_tree


class _Recording:
    """Remembers each answer, so scoring and the reply cost one model call between them."""

    name = "recording"

    def __init__(self, brain: Brain) -> None:
        self._brain = brain
        self._answers: dict[tuple[str, str], BrainResult] = {}
        self.last_brain = ""
        self.last_seconds = 0.0

    def understand(self, message: str, ctx: Context) -> BrainResult:
        key = (message, repr(ctx))
        if key not in self._answers:
            start = time.perf_counter()
            self._answers[key] = self._brain.understand(message, ctx)
            self.last_seconds = time.perf_counter() - start
            self.last_brain = self._answers[key].brain
        return self._answers[key]


def _checklist_summary(checklist: Checklist | None) -> list[dict[str, Any]]:
    if checklist is None:
        return []
    return [
        {
            "country": group.country,
            "purpose": group.purpose,
            "relation": person.relation,
            "profession": person.profession,
            "documents": [d.id for d in person.documents],
            "placeholders": [d.id for d in person.documents if "[PLACEHOLDER]" in d.label],
        }
        for group in checklist.groups
        for person in group.travellers
    ]


def _reply_for(
    case: Case, engine: ChatEngine, tree: LoadedTree
) -> tuple[BotReply, Checklist | None]:
    ctx = case.context
    state = ChatState(
        form=ctx.form,
        stage=ctx.stage,
        asked_field=ctx.asked,
        asked_travellers=ctx.asked_travellers,
        tree_version=tree.version,
    )
    state, reply = engine.handle(state, text=case.message)
    if reply.stage == "confirming":
        return reply, engine.handle(state, choice="confirm")[1].checklist
    return reply, reply.checklist


def run(
    brain: Brain, cases: Sequence[Case], tree: LoadedTree, ask_age: frozenset[str]
) -> dict[str, Any]:
    recording = _Recording(brain)
    engine = ChatEngine(lambda: tree, recording, ask_age_for=ask_age)
    outcomes, rows = [], []
    for case in cases:
        recording.last_seconds = 0.0
        outcome = run_case(recording, case, tree)
        answered_by, seconds = recording.last_brain, recording.last_seconds
        reply, checklist = _reply_for(case, engine, tree)
        outcomes.append(outcome)
        rows.append(
            {
                "id": case.id,
                "message": case.message,
                "exact": outcome.exact,
                "wrong": [f for f in FIELDS if not outcome.matches(f)],
                "brain": answered_by,
                "confident": outcome.confident,
                "seconds": round(seconds, 2),
                "got": outcome.form.model_dump(exclude_defaults=True),
                "want": case.expect.form.model_dump(exclude_defaults=True),
                "reply": reply.text,
                "buttons": [b.label for b in reply.buttons],
                "checklist": _checklist_summary(checklist),
            }
        )
    return {"scores": score(outcomes), "rows": rows}


def compare(folder: Path) -> str:
    """A readable report of every run in ``folder``: scores, then message by message."""
    runs = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(folder.glob("*.json"))
    }
    if not runs:
        raise SystemExit(f"no runs in {folder}")
    labels = list(runs)
    lines = [
        "# Trial results",
        "",
        "| run | " + " | ".join(next(iter(runs.values()))["scores"]) + " |",
    ]
    lines.append("|---" * (len(next(iter(runs.values()))["scores"]) + 1) + "|")
    for label, data in runs.items():
        values = " | ".join(f"{v:.0%}" for v in data["scores"].values())
        lines.append(f"| {label} | {values} |")

    lines += ["", "## Message by message", "", "| id | message | " + " | ".join(labels) + " |"]
    lines.append("|---" * (len(labels) + 2) + "|")
    for i, row in enumerate(next(iter(runs.values()))["rows"]):
        marks = [
            "ok" if data["rows"][i]["exact"] else ", ".join(data["rows"][i]["wrong"])
            for data in runs.values()
        ]
        message = row["message"].replace("|", "/")
        lines.append(f"| {row['id']} | {message} | " + " | ".join(marks) + " |")

    for label, data in runs.items():
        lines += ["", f"## {label}: replies and misses", ""]
        for row in data["rows"]:
            mark = "ok" if row["exact"] else "MISS"
            lines.append(f"### {row['id']} [{mark}] {row['message']}")
            lines.append(f"- answered by {row['brain'] or '-'} in {row['seconds']}s")
            if not row["exact"]:
                lines.append(f"- wrong: {', '.join(row['wrong'])}")
                lines.append(f"- got: `{json.dumps(row['got'], ensure_ascii=False)}`")
                lines.append(f"- want: `{json.dumps(row['want'], ensure_ascii=False)}`")
            lines.append(f"- bot said: {row['reply'].replace(chr(10), ' / ')}")
            for person in row["checklist"]:
                held = f", placeholders: {person['placeholders']}" if person["placeholders"] else ""
                lines.append(
                    f"- checklist: {person['relation']} ({person['profession']}, "
                    f"{person['country']}/{person['purpose']}): "
                    f"{len(person['documents'])} documents{held}"
                )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trial", description=__doc__)
    p.add_argument("--brains", default=None)
    p.add_argument("--solo", action="store_true", help="the brains without rules first")
    p.add_argument("--set", type=Path, default=DEFAULT_SET, dest="case_set")
    p.add_argument("--label", default="run")
    p.add_argument("--out", type=Path, default=Path("trial-out"))
    p.add_argument("--compare", type=Path, default=None, help="write comparison.md for a folder")
    args = p.parse_args(argv)

    if args.compare is not None:
        (args.compare / "comparison.md").write_text(compare(args.compare), encoding="utf-8")
        print(f"wrote {args.compare / 'comparison.md'}")
        return 0

    tree = load_tree(default_tree_path())
    lexicon = load_lexicon(default_lexicon_path(), tree)
    names = brain_names(args.brains, with_rules=not args.solo)
    brain = build_cascade(lexicon, names, tree=lambda: tree)
    result = run(brain, load_cases(args.case_set), tree, lexicon.ask_age_for)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.label}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    summary = "  ".join(f"{k} {v:.0%}" for k, v in result["scores"].items())
    print(f"{args.label}: {' -> '.join(names)}  {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
