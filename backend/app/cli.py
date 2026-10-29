"""Command-line front end for the resolver (Phase 1: works without any AI).

Example::

    uv run gvs-resolve --country japan --purpose tourism \\
        -t relation=self,profession=employed -t relation=mother,profession=homemaker \\
        -t relation=son,age=3
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from app.form import MAX_TRAVELLERS, Traveller, TripForm
from app.paths import default_tree_path
from app.resolver import Checklist, FormError, IncompleteFormError, resolve
from app.tree import TreeError, load_tree

_TRAVELLER_KEYS = frozenset({"relation", "profession", "age", "country", "purpose"})


def parse_traveller(spec: str) -> Traveller:
    """Parse ``key=value,key=value`` into a :class:`Traveller`."""
    fields: dict[str, str | int] = {}
    for part in spec.split(","):
        key, sep, value = part.strip().partition("=")
        if not sep or key not in _TRAVELLER_KEYS or not value:
            raise argparse.ArgumentTypeError(
                f"bad traveller field {part!r}; use key=value with keys {sorted(_TRAVELLER_KEYS)}"
            )
        if key in fields:
            raise argparse.ArgumentTypeError(f"{key!r} given twice in {spec!r}")
        if key == "age":
            if not value.isdecimal():
                raise argparse.ArgumentTypeError(f"age must be a whole number, got {value!r}")
            fields[key] = int(value)
        else:
            fields[key] = value
    try:
        return Traveller.model_validate(fields)
    except ValidationError as exc:
        raise argparse.ArgumentTypeError(f"invalid traveller {spec!r}: {exc}") from exc


def render(checklist: Checklist, *, disclaimer: bool = True) -> str:
    """Plain-text checklist. The chat reply already carries the disclaimer, so the
    REPL passes ``disclaimer=False`` to avoid printing it twice."""
    lines: list[str] = []
    for g in checklist.groups:
        reviewed = g.last_reviewed.isoformat() if g.last_reviewed else "not yet reviewed"
        lines.append(f"== {g.country_name} - {g.purpose_label}  (last reviewed: {reviewed})")
        if g.shared:
            lines.append("  Shared, once per application:")
            lines += [f"    - {d.label}" for d in g.shared]
        for p in g.travellers:
            age = f", age {p.age}" if p.age is not None else ""
            lines.append(f"  {p.index + 1}. {p.relation_label} ({p.profession_label}{age})")
            lines += [f"    - {d.label}" for d in p.documents]
    lines.append("")
    if disclaimer:
        lines.append("Confirm with the embassy or VFS before applying.")
    if checklist.tree_status != "reviewed":
        lines.append(f"WARNING: document data status is '{checklist.tree_status}'.")
    lines.append(f"(tree {checklist.tree_version})")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gvs-resolve", description="Print a visa document checklist.")
    p.add_argument("--country", required=True)
    p.add_argument("--purpose", required=True)
    p.add_argument(
        "-t",
        "--traveller",
        action="append",
        type=parse_traveller,
        required=True,
        help="relation=..,profession=..,age=..,country=..,purpose=.. (repeatable)",
    )
    p.add_argument("--tree", type=Path, default=None, help="path to visa-tree.json")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.traveller) > MAX_TRAVELLERS:
        print(f"error: at most {MAX_TRAVELLERS} travellers", file=sys.stderr)
        return 2
    try:
        tree = load_tree(args.tree or default_tree_path())
        form = TripForm(country=args.country, purpose=args.purpose, travellers=args.traveller)
        checklist = resolve(form, tree)
    except TreeError as exc:
        print(f"error: visa-tree.json is invalid: {exc}", file=sys.stderr)
        return 1
    except IncompleteFormError as exc:
        gaps = [
            m.field if m.traveller is None else f"{m.field}[{m.traveller}]" for m in exc.missing
        ]
        print(f"error: missing {', '.join(gaps)}", file=sys.stderr)
        return 2
    except (FormError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(checklist.model_dump_json(indent=2) if args.json else render(checklist))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
