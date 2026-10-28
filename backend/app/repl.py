"""Talk to the bot in a terminal: ``uv run gvs-chat``.

Type a message, or a button's number to tap it. ``/state`` shows the stored
form (what would be saved), ``/quit`` exits.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from app.brains.factory import brain_names, build_cascade
from app.brains.lexicon import load_lexicon
from app.chat import BotReply, ChatEngine
from app.cli import render
from app.paths import default_lexicon_path, default_tree_path
from app.tree import TreeStore


def _show(reply: BotReply) -> None:
    print(f"\nbot> {reply.text.replace(chr(10), chr(10) + '     ')}")
    if reply.checklist is not None:
        print("\n" + render(reply.checklist))
    for n, b in enumerate(reply.buttons, 1):
        print(f"     [{n}] {b.label}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gvs-chat", description="Chat with the bot in a terminal.")
    p.add_argument("--brains", default=None, help="e.g. rules (default from $GVS_BRAINS)")
    args = p.parse_args(argv)

    store = TreeStore(default_tree_path())
    lexicon = load_lexicon(default_lexicon_path(), store.get())
    names = brain_names(args.brains)
    engine = ChatEngine(store.get, build_cascade(lexicon, names), ask_age_for=lexicon.ask_age_for)
    print(f"brains: {' -> '.join(names)}   (/state, /quit)")

    state, reply = engine.start()
    _show(reply)
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line == "/quit":
            return 0
        if line == "/state":
            print(state.model_dump_json(indent=2))
            continue
        if line.isdigit() and 1 <= int(line) <= len(reply.buttons):
            state, reply = engine.handle(state, choice=reply.buttons[int(line) - 1].choice)
        else:
            state, reply = engine.handle(state, text=line)
        _show(reply)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
