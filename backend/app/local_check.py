"""Check the local model setup end to end: ``uv run gvs-local-check``.

Reads the same ``GVS_LOCAL_*`` settings as the bot, confirms Ollama answers and
has the model, then sends one probe message twice (cold, then warm) and checks
the form it fills.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from collections.abc import Sequence
from typing import Final

from pydantic import ValidationError

from app.brains import BrainUnavailable
from app.brains.local import LocalBrain, LocalSettings, Transport, http_transport
from app.extraction import Context
from app.form import TripForm
from app.paths import default_tree_path
from app.tree import LoadedTree, load_tree

# Not one of the prompt's examples, so a model that only copies them fails.
PROBE: Final = "my parents are both retired and want to go to London for a holiday"
COLD_TIMEOUT: Final = 120.0
EXPECT: Final = ("uk", "tourism", ["father", "mother"])


def check(
    tree: LoadedTree, settings: LocalSettings, *, transport: Transport = http_transport
) -> list[str]:
    """Problems with the local setup, or ``[]``. Prints progress as it goes."""
    print(f"model server: {settings.url}   model: {settings.model}")
    try:
        tags = json.loads(transport(f"{settings.url}/api/tags", None, settings.timeout))
    except (urllib.error.URLError, OSError, ValueError, BrainUnavailable) as exc:
        return [f"cannot reach Ollama at {settings.url} ({type(exc).__name__}); is it running?"]
    models = tags.get("models", []) if isinstance(tags, dict) else []
    names = {m.get("name") for m in models if isinstance(m, dict)}
    wanted = settings.model if ":" in settings.model else f"{settings.model}:latest"
    if wanted not in names:
        have = ", ".join(sorted(str(n) for n in names)) or "none"
        return [
            f"model {settings.model!r} is not pulled (have: {have}); "
            f"run: ollama pull {settings.model}"
        ]
    if not settings.is_loopback:
        print("warning: the model server is not on this machine; Ollama has no login")

    # The cold run loads the model from disk into VRAM, which can take a while.
    loading = settings.model_copy(update={"timeout": max(settings.timeout, COLD_TIMEOUT)})
    ctx = Context(form=TripForm(), asked=None, stage="collecting")
    problems: list[str] = []
    for attempt, s in (("cold", loading), ("warm", settings)):
        brain = LocalBrain(lambda: tree, s, transport=transport)
        started = time.perf_counter()
        try:
            ex = brain.understand(PROBE, ctx).extraction
        except BrainUnavailable as exc:
            return [f"{attempt} run failed: {exc}"]
        took = time.perf_counter() - started
        print(f"{attempt} run: {took:.2f}s")
    relations = sorted(p.relation or "" for p in ex.travellers)
    if (ex.country, ex.purpose, relations) != EXPECT:
        problems.append(f"probe understood wrongly: {ex.model_dump(exclude_defaults=True)}")
    if took > settings.timeout / 2:
        problems.append(f"warm run took {took:.1f}s; check the model is on the GPU (ollama ps)")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gvs-local-check", description="Check the local model setup.")
    p.parse_args(argv)
    try:
        settings = LocalSettings.from_env()
    except (ValueError, ValidationError) as exc:
        print(f"FAIL bad GVS_LOCAL_* setting: {exc}")
        return 2
    problems = check(load_tree(default_tree_path()), settings)
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print("OK   local model is ready: set GVS_BRAINS=rules,local")
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
