"""Score a brain on a fixed set of messages: ``uv run gvs-eval --brains rules``.

Each case gives a starting form and the expected form *after* the message is
merged, so the score measures what users see, not raw extraction. Scores are
per field and exact-match. Model brains vary run to run, so use ``--runs 3``
for them; the rules brain is deterministic.

These are development regression sets written by us, not a guarantee on
arbitrary real-world messages.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.brains import Brain, BrainUnavailable
from app.brains.factory import brain_names, build_cascade
from app.brains.lexicon import load_lexicon
from app.chat import merge
from app.extraction import Asked, AskedField, Context, Intent, Stage, sanitize
from app.form import TripForm
from app.paths import default_lexicon_path, default_tree_path
from app.tree import LoadedTree, load_tree

DEFAULT_SET = Path(__file__).resolve().parents[1] / "evals" / "understanding.jsonl"
FIELDS = ("country", "purpose", "travellers", "expected_count", "intent", "on_topic")


class CaseContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    form: TripForm = TripForm()
    asked: AskedField = "open"
    asked_travellers: tuple[int, ...] = ()
    stage: Stage = "collecting"


class Expect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    form: TripForm
    intent: Intent = "give_details"
    on_topic: bool = True


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    message: str
    context: CaseContext = CaseContext()
    expect: Expect


@dataclass(frozen=True, slots=True)
class Outcome:
    case: Case
    form: TripForm
    intent: str
    on_topic: bool
    confident: bool

    def matches(self, field: str) -> bool:
        exp = self.case.expect
        if field == "intent":
            return self.intent == exp.intent
        if field == "on_topic":
            return self.on_topic == exp.on_topic
        if field == "travellers":  # the same people in any order is the same checklist
            return _people(self.form) == _people(exp.form)
        return bool(getattr(self.form, field) == getattr(exp.form, field))

    @property
    def exact(self) -> bool:
        return all(self.matches(f) for f in FIELDS)


def _people(form: TripForm) -> list[str]:
    return sorted(t.model_dump_json() for t in form.travellers)


def load_cases(path: Path) -> list[Case]:
    cases = [
        Case.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    ]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError(f"{path}: duplicate case ids")
    return cases


def run_case(brain: Brain, case: Case, tree: LoadedTree) -> Outcome:
    ctx_in = case.context
    ctx = Context(ctx_in.form, Asked(ctx_in.asked, ctx_in.asked_travellers), ctx_in.stage)
    try:
        result = brain.understand(case.message, ctx)
    except BrainUnavailable:
        return Outcome(case, ctx_in.form, "unavailable", True, False)
    ex = sanitize(result.extraction, tree, len(ctx_in.form.travellers))
    overwrite = ex.intent == "correct" or ctx_in.stage != "collecting"
    form, _ = merge(ctx_in.form, ex, overwrite=overwrite)
    return Outcome(case, form, ex.intent, ex.on_topic, result.confident)


def score(outcomes: Sequence[Outcome]) -> dict[str, float]:
    n = len(outcomes) or 1
    scores = {f: sum(o.matches(f) for o in outcomes) / n for f in FIELDS}
    scores["exact"] = sum(o.exact for o in outcomes) / n
    scores["confident"] = sum(o.confident for o in outcomes) / n
    return scores


def _diff(o: Outcome) -> str:
    wrong = [f for f in FIELDS if not o.matches(f)]
    got: dict[str, Any] = {
        "intent": o.intent,
        "on_topic": o.on_topic,
        **o.form.model_dump(exclude_defaults=True),
    }
    want: dict[str, Any] = {
        "intent": o.case.expect.intent,
        "on_topic": o.case.expect.on_topic,
        **o.case.expect.form.model_dump(exclude_defaults=True),
    }
    return (
        f"  x {o.case.id}: {o.case.message!r}\n    wrong: {', '.join(wrong)}\n"
        f"    got:  {json.dumps(got)}\n    want: {json.dumps(want)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gvs-eval", description="Score brains on a message set.")
    p.add_argument("--brains", default=None, help="cascade to test, e.g. rules or rules,haiku")
    p.add_argument("--set", type=Path, default=DEFAULT_SET, dest="case_set")
    p.add_argument("--runs", type=int, default=1, help="repeat runs (for model brains)")
    p.add_argument("--min-exact", type=float, default=0.0, help="fail below this exact rate")
    p.add_argument("--show", type=int, default=20, help="how many failures to print")
    args = p.parse_args(argv)

    tree = load_tree(default_tree_path())
    lexicon = load_lexicon(default_lexicon_path(), tree)
    names = brain_names(args.brains)
    brain = build_cascade(lexicon, names)
    cases = load_cases(args.case_set)

    outcomes = [run_case(brain, c, tree) for _ in range(max(1, args.runs)) for c in cases]
    scores = score(outcomes)
    print(f"brains: {' -> '.join(names)}   cases: {len(cases)} x {args.runs} run(s)")
    for name, value in scores.items():
        print(f"  {name:<15} {value:7.1%}")
    failures = [o for o in outcomes if not o.exact]
    by_case = Counter(o.case.id for o in failures)
    for o in {o.case.id: o for o in failures}.values():
        if args.show <= 0:
            break
        print(_diff(o) + (f"  ({by_case[o.case.id]}/{args.runs} runs)" if args.runs > 1 else ""))
        args.show -= 1
    return 0 if scores["exact"] >= args.min_exact else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
