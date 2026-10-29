"""The conversation: one small form per chat, filled until complete.

``ChatEngine.handle`` is a pure step function: (state, message or button) ->
(new state, reply). It never stores raw user text; the state holds only ids
and numbers, so it is safe to persist and log.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from app import templates as T
from app.brains import Brain, BrainUnavailable
from app.extraction import Asked, AskedField, Context, Extraction, Stage, sanitize
from app.form import MAX_TRAVELLERS, Traveller, TripForm, unknown_ids
from app.resolver import Checklist, effective_profession, missing_fields, resolve
from app.templates import Button
from app.text import MAX_MESSAGE_CHARS, MessageTooLong, clean, has_devanagari
from app.tree import LoadedTree

log = logging.getLogger(__name__)

MAX_TURNS: Final = 20
MISSES_BEFORE_BUTTONS: Final = 2
_FIELDS: Final = ("profession", "age", "country", "purpose")
_CHOICE: Final = re.compile(r"^[a-z]+(?::[a-z0-9_-]{1,40}){0,2}$")


class ChatState(BaseModel):
    """Everything the server keeps per chat. No free text, no personal data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    form: TripForm = TripForm()
    stage: Stage = "collecting"
    asked_field: AskedField = "open"
    asked_travellers: tuple[int, ...] = ()
    misses: int = Field(default=0, ge=0)
    turns: int = Field(default=0, ge=0)
    tree_version: str = ""

    @property
    def asked(self) -> Asked:
        return Asked(self.asked_field, self.asked_travellers)


class BotReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    buttons: tuple[Button, ...] = ()
    checklist: Checklist | None = None
    stage: Stage = "collecting"


# --------------------------------------------------------------------------- merging


def _conflicts(current: dict[str, Any], patch: Any) -> bool:
    return any(
        getattr(patch, f) is not None
        and current.get(f) is not None
        and getattr(patch, f) != current[f]
        for f in _FIELDS
    )


def merge(form: TripForm, ex: Extraction, *, overwrite: bool) -> tuple[TripForm, bool]:
    """Apply an extraction to the form. Returns (new form, hit traveller limit).

    New values fill gaps; existing values change only when ``overwrite``
    (a correction, or any edit after the summary was shown).
    """
    country = ex.country if ex.country and (form.country is None or overwrite) else form.country
    purpose = ex.purpose if ex.purpose and (form.purpose is None or overwrite) else form.purpose
    people: list[dict[str, Any]] = [t.model_dump(exclude_none=True) for t in form.travellers]
    used: set[int] = set()
    removals: set[int] = set()
    overflow = False
    added_other = False
    added_relations: set[str] = set()

    for patch in ex.travellers:
        idx: int | None = patch.ref if patch.ref is not None and patch.ref < len(people) else None
        if idx is None and patch.relation is not None:
            candidates = [
                i for i, t in enumerate(people) if t["relation"] == patch.relation and i not in used
            ]
            if patch.relation == "self":
                candidates = [i for i, t in enumerate(people) if t["relation"] == "self"]
            idx = next((i for i in candidates if not _conflicts(people[i], patch)), None)
            if idx is None and candidates and (overwrite or patch.relation == "self"):
                idx = candidates[0]
        if idx is None:
            if patch.relation is None or patch.remove:
                continue
            if len(people) >= MAX_TRAVELLERS:
                overflow = True
                continue
            people.append({"relation": patch.relation})
            idx = len(people) - 1
            added_other = added_other or patch.relation != "self"
            added_relations.add(patch.relation)
        used.add(idx)
        if patch.remove:
            removals.add(idx)
            continue
        person = people[idx]
        for f in _FIELDS:
            value = getattr(patch, f)
            if value is None:
                continue
            if f in ("country", "purpose") and value == (country if f == "country" else purpose):
                if overwrite:
                    person.pop(f, None)  # "mom is going to Japan too" clears her override
                continue
            if person.get(f) is None or overwrite:
                person[f] = value

    people = [t for i, t in enumerate(people) if i not in removals]
    expected = form.expected_count
    if ex.traveller_count is not None and ex.traveller_count > MAX_TRAVELLERS:
        overflow = True
    elif ex.traveller_count is not None and (
        expected is None or overwrite or ex.traveller_count > expected
    ):
        expected = ex.traveller_count
    if expected is not None and expected <= len(people):
        expected = None
    # "with my family": wait for names; naming anyone (or "that's all") settles it.
    pending = (form.others_pending or ex.others_coming) and not added_other
    # "with my friends": ask how many next; "3 friends" or "that's all" settles it.
    group = ex.group_of or form.pending_group
    if ex.group_of is None and group in added_relations:
        group = None
    if ex.traveller_count is not None and ex.traveller_count <= len(people):
        pending = False
        group = None
    if group is not None and len(people) >= MAX_TRAVELLERS:
        overflow = overflow or ex.group_of is not None
        group = None
    new = TripForm(
        country=country,
        purpose=purpose,
        travellers=tuple(Traveller(**t) for t in people),
        expected_count=expected,
        others_pending=pending and bool(people),
        pending_group=group,
    )
    return new, overflow


def prune(form: TripForm, tree: LoadedTree) -> TripForm:
    """Drop values that a reloaded tree no longer defines."""
    if not unknown_ids(form, tree):
        return form
    d = tree.data
    people = [
        Traveller(
            relation=t.relation,
            profession=t.profession if t.profession in d.professions else None,
            age=t.age,
            country=t.country if t.country in d.countries else None,
            purpose=t.purpose if t.purpose in d.purposes else None,
        )
        for t in form.travellers
        if t.relation in d.relations
    ]
    return TripForm(
        country=form.country if form.country in d.countries else None,
        purpose=form.purpose if form.purpose in d.purposes else None,
        travellers=tuple(people),
        expected_count=form.expected_count,
        others_pending=form.others_pending,
        pending_group=form.pending_group if form.pending_group in d.relations else None,
    )


def _last_answer(state: ChatState, still_missing: tuple[int, ...]) -> str | None:
    """The job just given for one of several people asked together, if the rest
    are still missing it: offered as a one-tap 'Same' for all of them."""
    if state.asked_field != "profession" or len(state.asked_travellers) < 2:
        return None
    people = state.form.travellers
    asked = [i for i in state.asked_travellers if i < len(people)]
    answered = [i for i in asked if people[i].profession is not None]
    if not answered or not set(still_missing) <= set(asked):
        return None
    return people[answered[-1]].profession


# --------------------------------------------------------------------------- engine


class ChatEngine:
    def __init__(
        self,
        tree: Callable[[], LoadedTree],
        brain: Brain,
        *,
        ask_age_for: frozenset[str] = frozenset({"son", "daughter", "child"}),
        max_turns: int = MAX_TURNS,
        max_chars: int = MAX_MESSAGE_CHARS,
    ) -> None:
        self._tree = tree
        self._brain = brain
        self._ask_age_for = ask_age_for
        self._max_turns = max_turns
        self._max_chars = max_chars

    # ---------------------------------------------------------------- entry points

    def start(self) -> tuple[ChatState, BotReply]:
        return ChatState(tree_version=self._tree().version), BotReply(text=T.OPENING)

    def handle(
        self, state: ChatState, *, text: str | None = None, choice: str | None = None
    ) -> tuple[ChatState, BotReply]:
        tree = self._tree()
        if state.tree_version != tree.version:
            state = state.model_copy(
                update={"form": prune(state.form, tree), "tree_version": tree.version}
            )
        if choice == "restart":
            return self.start()
        if state.turns >= self._max_turns:
            return state, BotReply(text=T.TURN_LIMIT, buttons=(T.START_OVER,), stage=state.stage)
        state = state.model_copy(update={"turns": state.turns + 1})

        if choice is not None:
            return self._choice(state, choice, tree)
        try:
            message = clean(text or "", self._max_chars)
        except MessageTooLong:
            prefix = T.TOO_LONG.format(limit=self._max_chars)
            return self._ask_next(state, tree, prefix=prefix, progressed=True)
        if not message:
            return self._ask_next(state, tree, prefix=None, progressed=True)
        return self._message(state, message, tree)

    # ---------------------------------------------------------------- free text

    def _message(
        self, state: ChatState, message: str, tree: LoadedTree
    ) -> tuple[ChatState, BotReply]:
        ctx = Context(form=state.form, asked=state.asked, stage=state.stage)
        try:
            result = self._brain.understand(message, ctx)
        except BrainUnavailable:
            log.exception("all brains failed; falling back to buttons")
            state = state.model_copy(update={"misses": MISSES_BEFORE_BUTTONS})
            return self._ask_next(state, tree, prefix=T.NOT_UNDERSTOOD, progressed=False)
        ex = sanitize(result.extraction, tree, len(state.form.travellers))
        log.info("brain=%s confident=%s intent=%s", result.brain, result.confident, ex.intent)

        if ex.intent == "restart":
            return self.start()
        if not ex.on_topic:
            empty = state.form == TripForm()
            if empty and state.misses + 1 < MISSES_BEFORE_BUTTONS:
                state = state.model_copy(update={"misses": state.misses + 1})
                return state, BotReply(text=T.OFF_TOPIC_START, stage=state.stage)
            return self._ask_next(state, tree, prefix=T.OFF_TOPIC, progressed=False)

        if state.stage in ("confirming", "done") and not ex.has_details:
            if ex.intent == "confirm":
                return (
                    self._finish(state, tree)
                    if state.stage == "confirming"
                    else (
                        state,
                        BotReply(text=T.ALREADY_DONE, buttons=(T.START_OVER,), stage="done"),
                    )
                )
            if ex.intent == "correct":
                state = state.model_copy(update={"asked_field": "open", "asked_travellers": ()})
                return state, BotReply(
                    text=T.WHAT_CHANGE, buttons=(T.START_OVER,), stage=state.stage
                )

        # Things to say before the next question; each is a fixed template.
        site = tree.data.meta.site_url
        notes: list[str] = []
        if ex.asks_help:
            notes.append(T.help_text(tree))
        if ex.other_country and not ex.country:
            other = tree.data.other_countries[ex.other_country]
            notes.append(T.unsupported(other.name, ex.other_country, site))
        if ex.asks_office:
            notes.append(T.office(site))

        overwrite = ex.intent == "correct" or state.stage != "collecting"
        form, overflow = merge(state.form, ex, overwrite=overwrite)
        progressed = form != state.form
        if overflow:
            notes.append(T.TOO_MANY.format(limit=MAX_TRAVELLERS))
        quiet = not progressed and not notes
        opener = quiet and state.form == TripForm() and not has_devanagari(message)
        if opener and state.misses + 1 < MISSES_BEFORE_BUTTONS:
            # "hi", "hello", "hmm": nothing to fill yet, so just open again.
            state = state.model_copy(update={"misses": state.misses + 1})
            return state, BotReply(text=T.OPENING, stage=state.stage)
        state = state.model_copy(update={"form": form})
        if quiet:
            notes.append(T.HINDI_SCRIPT if has_devanagari(message) else T.NOT_UNDERSTOOD)
        return self._ask_next(
            state,
            tree,
            prefix=" ".join(notes),
            progressed=not quiet,
            purpose_choices=ex.purpose_choices,
        )

    # ---------------------------------------------------------------- buttons

    def _choice(
        self, state: ChatState, choice: str, tree: LoadedTree
    ) -> tuple[ChatState, BotReply]:
        form = state.form
        d = tree.data
        parts = choice.split(":") if _CHOICE.fullmatch(choice) else ["invalid"]
        kind, args = parts[0], parts[1:]
        new: TripForm | None = None

        if kind == "confirm" and state.stage == "confirming":
            return self._finish(state, tree)
        if kind == "country" and len(args) == 1 and args[0] in d.countries:
            new = form.model_copy(update={"country": args[0]})
        elif kind == "purpose" and len(args) == 1 and args[0] in d.purposes:
            new = form.model_copy(update={"purpose": args[0]})
        elif kind == "solo" and not form.travellers:
            new = form.model_copy(update={"travellers": (Traveller(relation="self"),)})
        elif kind == "add" and len(args) == 1 and args[0] in d.relations:
            has_self = any(t.relation == "self" for t in form.travellers)
            if len(form.travellers) < MAX_TRAVELLERS and not (args[0] == "self" and has_self):
                new = form.model_copy(
                    update={"travellers": (*form.travellers, Traveller(relation=args[0]))}
                )
        elif kind == "everyone" and form.travellers:
            new = form.model_copy(
                update={"expected_count": None, "others_pending": False, "pending_group": None}
            )
        elif (
            kind == "group"
            and len(args) == 1
            and args[0].isdigit()
            and form.pending_group is not None
            and 1 <= int(args[0]) <= MAX_TRAVELLERS - len(form.travellers)
        ):
            group = tuple(Traveller(relation=form.pending_group) for _ in range(int(args[0])))
            new = form.model_copy(
                update={"travellers": (*form.travellers, *group), "pending_group": None}
            )
        elif (
            kind == "prof"
            and len(args) == 2
            and args[0].isdigit()
            and int(args[0]) < len(form.travellers)
            and args[1] in d.professions
        ):
            people = list(form.travellers)
            i = int(args[0])
            people[i] = people[i].model_copy(update={"profession": args[1]})
            new = form.model_copy(update={"travellers": tuple(people)})
        elif (
            kind == "same"
            and len(args) == 1
            and args[0] in d.professions
            and state.asked_field == "profession"
        ):
            people = list(form.travellers)
            for i in state.asked_travellers:
                if i < len(people) and people[i].profession is None:
                    people[i] = people[i].model_copy(update={"profession": args[0]})
            new = form.model_copy(update={"travellers": tuple(people)})

        if new is None:  # stale or forged button: ignore it, re-ask
            return self._ask_next(state, tree, prefix=None, progressed=False)
        state = state.model_copy(update={"form": TripForm.model_validate(new.model_dump())})
        if state.stage != "collecting":
            state = state.model_copy(update={"stage": "collecting"})
        return self._ask_next(state, tree, prefix=None, progressed=True)

    # ---------------------------------------------------------------- questions

    def _ask_next(
        self,
        state: ChatState,
        tree: LoadedTree,
        *,
        prefix: str | None,
        progressed: bool,
        purpose_choices: tuple[str, ...] = (),
    ) -> tuple[ChatState, BotReply]:
        misses = 0 if progressed else state.misses + 1
        form = state.form
        missing = missing_fields(form, tree)
        show = misses >= MISSES_BEFORE_BUTTONS
        buttons: tuple[Button, ...] = ()
        asked: tuple[AskedField, tuple[int, ...]]
        stage: Stage

        if not missing:
            professions = [
                effective_profession(t, form.country_of(t), tree) or "other"
                for t in form.travellers
            ]
            text = T.summary(form, professions, tree)
            asked, stage = ("confirm", ()), "confirming"
            buttons = (T.CONFIRM, T.START_OVER)
        else:
            stage = "collecting"
            gap = missing[0]
            if gap.field == "country":
                asked, text, buttons = ("country", ()), T.ASK_COUNTRY, T.country_buttons(tree)
            elif gap.field == "purpose":
                asked = ("purpose", ())
                if len(purpose_choices) > 1:
                    text = T.ask_main_purpose(purpose_choices, tree)
                    buttons = T.purpose_buttons(tree, only=purpose_choices)
                else:
                    text, buttons = T.ASK_PURPOSE, T.purpose_buttons(tree)
            elif gap.field == "travellers" and form.pending_group is not None:
                asked = ("group_size", ())
                text = T.ask_group_size(form.pending_group, tree)
                buttons = T.group_size_buttons(MAX_TRAVELLERS - len(form.travellers))
            elif gap.field == "travellers":
                more = bool(form.travellers)
                text = T.ask_more(form, tree) if more else T.ASK_WHO
                asked = ("travellers", ())
                buttons = T.who_buttons(form, tree) if show or form.others_pending else ()
            else:
                gaps = tuple(m.traveller for m in missing if m.traveller is not None)
                kids = tuple(
                    i
                    for i in gaps
                    if form.travellers[i].relation in self._ask_age_for
                    and form.travellers[i].age is None
                )
                adults = tuple(i for i in gaps if i not in kids)
                if adults:
                    asked = ("profession", adults)
                    text = T.ask_profession(form, adults, tree)
                    buttons = T.profession_buttons(adults[0], tree) if show else ()
                    if same := _last_answer(state, adults):
                        buttons = (T.same_button(same, tree), *buttons)
                else:
                    asked = ("age", kids)
                    text = T.ask_age(form, kids, tree)
                    buttons = T.child_buttons(kids[0], tree) if show else ()
            if show and gap.field not in ("country", "purpose"):
                text = f"{text}\n{T.BUTTONS_HINT}"

        if prefix:
            text = f"{prefix} {text}"
        state = state.model_copy(
            update={
                "asked_field": asked[0],
                "asked_travellers": asked[1],
                "misses": misses,
                "stage": stage,
            }
        )
        return state, BotReply(text=text, buttons=buttons, stage=stage)

    def _finish(self, state: ChatState, tree: LoadedTree) -> tuple[ChatState, BotReply]:
        checklist = resolve(state.form, tree)
        state = state.model_copy(
            update={"stage": "done", "asked_field": "open", "asked_travellers": (), "misses": 0}
        )
        text = T.result(checklist, tree.data.meta.site_url)
        return state, BotReply(
            text=text, buttons=(T.START_OVER,), checklist=checklist, stage="done"
        )
