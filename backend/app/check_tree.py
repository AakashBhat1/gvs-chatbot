"""Validate ``visa-tree.json``. Runs in CI on every commit.

Default mode fails on anything structurally wrong (unknown ids, bad keys,
duplicate keys, empty lists) and warns about unused documents and unreviewed
countries.

``--release`` is the go-live gate: it also fails unless the file is marked
``reviewed``, contains no placeholder text, and every country was reviewed
within ``--max-review-age-days``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from app.paths import default_tree_path
from app.tree import LoadedTree, TreeError, load_tree

PLACEHOLDER_MARK = "[PLACEHOLDER]"


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def audit(tree: LoadedTree, *, release: bool, max_review_age_days: int, today: date) -> Report:
    data = tree.data
    report = Report()

    used = {d for docs in data.rules.values() for d in docs}
    report.warnings += [f"document {d!r} is never used" for d in data.documents if d not in used]

    release_problems: list[str] = []
    for cid, country in data.countries.items():
        if country.last_reviewed is None:
            release_problems.append(f"country {cid!r} has never been reviewed")
        elif country.last_reviewed > today:
            release_problems.append(f"country {cid!r} has a review date in the future")
        elif (age := (today - country.last_reviewed).days) > max_review_age_days:
            release_problems.append(f"country {cid!r} last reviewed {age} days ago")
    if data.meta.status != "reviewed":
        release_problems.append(f"meta.status is {data.meta.status!r}, not 'reviewed'")
    placeholders = [d for d, doc in data.documents.items() if PLACEHOLDER_MARK in doc.label]
    if placeholders:
        release_problems.append(f"{len(placeholders)} documents still contain {PLACEHOLDER_MARK}")

    if release:
        report.errors += release_problems
    else:
        report.warnings += release_problems
    return report


def main(argv: Sequence[str] | None = None, *, today: date | None = None) -> int:
    p = argparse.ArgumentParser(prog="gvs-check-tree", description="Validate visa-tree.json.")
    p.add_argument("path", nargs="?", type=Path, default=None)
    p.add_argument("--release", action="store_true", help="enforce go-live requirements")
    p.add_argument("--max-review-age-days", type=int, default=180)
    args = p.parse_args(argv)

    path: Path = args.path or default_tree_path()
    try:
        tree = load_tree(path)
    except TreeError as exc:
        print(f"FAIL {path}", file=sys.stderr)
        for problem in str(exc).split("; "):
            print(f"  error: {problem}", file=sys.stderr)
        return 1

    report = audit(
        tree,
        release=args.release,
        max_review_age_days=args.max_review_age_days,
        today=today or date.today(),
    )
    for w in report.warnings:
        print(f"  warning: {w}", file=sys.stderr)
    for e in report.errors:
        print(f"  error: {e}", file=sys.stderr)
    data = tree.data
    summary = (
        f"{len(data.countries)} countries, {len(data.purposes)} purposes, "
        f"{len(data.professions)} professions, {len(data.relations)} relations, "
        f"{len(data.documents)} documents, {len(data.rules)} rules"
    )
    print(f"{'OK  ' if report.ok else 'FAIL'} {path} [{tree.version}] {summary}")
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
