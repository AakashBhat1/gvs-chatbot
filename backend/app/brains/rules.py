"""The no-AI brain: lexicon matching plus a few grammar rules.

Pipeline for one message::

    tokens -> spans (longest alias match) -> mentions (people) -> attach details
           -> trip-level country/purpose -> intent -> Extraction

Details attach to the nearest person mentioned *before* them in the same
sentence ("me doctor, mom lawyer, son 3"), or the next one after if there is
none ("3 year old son"). A bare answer with no person ("retired", "8") goes to
whoever the bot just asked about.

Confidence: the result is *confident* only if every word was explained (an
alias, a marker or a filler) and nothing was ambiguous. When it isn't, the
cascade can ask a smarter brain; with no smarter brain available, this result
is still used and the bot's follow-up questions fill the gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from app.brains import BrainResult
from app.brains.lexicon import Lexicon
from app.extraction import MAX_COUNT, Context, Extraction, Intent, TravellerPatch
from app.form import MAX_TRAVELLERS
from app.hindi import romanize
from app.text import has_devanagari, tokenize

HARD_SEPARATORS: Final = frozenset({".", ";", "!", "?"})
SOFT_SEPARATORS: Final = frozenset({",", "+"})
HINDI_IN: Final = frozenset({"me", "mai", "main"})  # "Japan me" = "in Japan"
HOST_PHRASE_ENDINGS: Final = frozenset({"my", "with", "a"})  # "visit my son" -> son is the host
NO_PROFESSION_SWAP: Final = "retired"  # "retired army officer" -> retired wins
# 'she' can't be a father or son; 'he' can't be a mother or daughter.
PRONOUN_EXCLUDES: Final = {
    "she": frozenset({"father", "son"}),
    "he": frozenset({"mother", "daughter"}),
}
WITH_BEFORE: Final = frozenset({"with"})  # "with my friends": "along with" ends in "with" too
WITH_AFTER: Final = frozenset({"saath", "sath"})  # "dost ke saath"
PERSON_KINDS: Final = frozenset(
    {
        "relation",
        "we",
        "solo",
        "couple",
        "parents",
        "two_of_us",
        "pronoun",
        "with_others",
        "one_of",
        "other_of",
    }
)


@dataclass(slots=True)
class Span:
    kind: str
    value: str
    tokens: tuple[str, ...]
    sentence: int = 0
    segment: int = 0
    used: bool = False
    fuzzy: bool = False  # matched by typo tolerance, so not fully certain

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


@dataclass(slots=True)
class Slot:
    relation: str | None = None
    ref: int | None = None
    profession: str | None = None
    age: int | None = None
    country: str | None = None
    purpose: str | None = None
    remove: bool = False
    host: bool = False
    from_plural: bool = False
    group: bool = False  # "my friends" with no number: reported as group_of, not a person


@dataclass(slots=True)
class Mention:
    pos: int  # index into spans
    slots: list[int]
    relation: str
    plural_pending: bool = False  # "kids" with no number: one slot per age that follows
    group: int = 0


@dataclass(slots=True)
class _Parse:
    spans: list[Span]
    slots: list[Slot] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    count: int | None = None
    singled: set[int] = field(default_factory=set)  # slots picked by "one of them"
    adults: int | None = None  # "two adults and one child" -> adults + children
    country: str | None = None
    purpose: str | None = None
    other_country: str | None = None
    purpose_choices: list[str] = field(default_factory=list)
    others_coming: bool = False
    group_of: str | None = None
    conflict: bool = False
    unexplained: int = 0
    ambiguous: int = 0


class RulesBrain:
    name = "rules"

    def __init__(self, lexicon: Lexicon) -> None:
        self.lex = lexicon

    # ------------------------------------------------------------------ public

    def understand(self, message: str, ctx: Context) -> BrainResult:
        p = _Parse(self._scan([self._from_script(t) for t in tokenize(message)]))
        self._mark_boundaries(p)
        self._make_mentions(p, ctx)
        self._speaker_goes_with(p)
        self._numbers(p, ctx)
        self._professions(p, ctx)
        self._places_and_purposes(p, ctx)
        self._removals(p)
        self._uncounted_groups(p, ctx)
        p.unexplained += sum(s.kind == "unknown" for s in p.spans)
        p.ambiguous += sum(s.fuzzy for s in p.spans)

        extraction = self._build(p, ctx, message)
        confident = p.unexplained == 0 and p.ambiguous == 0
        return BrainResult(extraction=extraction, confident=confident, brain=self.name)

    # ------------------------------------------------------------------ scanning

    def _from_script(self, token: str) -> str:
        """Hindi-script word -> the Hinglish spelling the lexicon knows (जापान -> japan).

        Words that appear in Hindi-script aliases are kept as they are, so exact
        matches (दुबई, संयुक्त अरब अमीरात) win over transliteration.
        """
        if token in self.lex.script_words or not has_devanagari(token):
            return token
        spellings = romanize(token)
        for spelling in spellings:
            if (spelling,) in self.lex.phrases or spelling in self.lex.numbers:
                return spelling
        return spellings[-1] if spellings else token

    def _scan(self, tokens: list[str]) -> list[Span]:
        lex = self.lex
        spans: list[Span] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in HARD_SEPARATORS or tok in SOFT_SEPARATORS:
                kind = "sep" if tok in HARD_SEPARATORS else "comma"
                spans.append(Span(kind, tok, (tok,)))
                i += 1
                continue
            for n in range(min(lex.max_phrase, len(tokens) - i), 0, -1):
                hit = lex.lookup(tuple(tokens[i : i + n]))
                if hit is not None:
                    spans.append(Span(hit[0], hit[1], tuple(tokens[i : i + n])))
                    i += n
                    break
            else:
                if tok.isdigit() and len(tok) <= 3:
                    spans.append(Span("number", tok, (tok,)))
                elif tok in lex.weak_numbers:
                    spans.append(Span("weak_number", str(lex.numbers[tok]), (tok,)))
                elif tok in lex.numbers:
                    spans.append(Span("number", str(lex.numbers[tok]), (tok,)))
                elif guess := lex.guess(tok):
                    spans.append(Span(guess[0], guess[1], (tok,), fuzzy=True))
                else:
                    spans.append(Span("unknown", tok, (tok,)))
                i += 1

        # Context fixes on the raw sequence, then drop fillers.
        for k, s in enumerate(spans):
            nxt = spans[k + 1] if k + 1 < len(spans) else None
            prev = _prev_meaningful(spans, k)
            if s.kind == "weak_number":
                is_count = nxt is not None and nxt.kind == "count_noun"
                s.kind = "number" if is_count else "filler"
            elif (
                s.kind == "relation"
                and s.tokens[0] in HINDI_IN
                and len(s.tokens) == 1
                and prev is not None
                and prev.kind in ("country", "purpose")
            ):
                s.kind = "filler"
        return [s for s in spans if s.kind != "filler"]

    @staticmethod
    def _mark_boundaries(p: _Parse) -> None:
        sentence = segment = 0
        for s in p.spans:
            if s.kind == "sep":
                sentence += 1
                segment += 1
            elif s.kind in ("comma", "conj"):
                segment += 1
            s.sentence, s.segment = sentence, segment

    # ------------------------------------------------------------------ people

    def _new_slots(self, p: _Parse, relation: str, n: int, *, plural: bool) -> list[int]:
        start = len(p.slots)
        for _ in range(n):
            if len(p.slots) >= MAX_COUNT:
                p.ambiguous += 1
                break
            p.slots.append(Slot(relation=relation, from_plural=plural))
        return list(range(start, len(p.slots)))

    def _self_slot(self, p: _Parse) -> list[int]:
        for i, slot in enumerate(p.slots):
            if slot.relation == "self":
                return [i]
        return self._new_slots(p, "self", 1, plural=False)

    def _make_mentions(self, p: _Parse, ctx: Context) -> None:
        group = 0
        last_kind = ""
        for k, s in enumerate(p.spans):
            if s.kind in PERSON_KINDS and last_kind != "person":
                group += 1
            created = self._mention_for(p, k, s, group, ctx)
            if created:
                last_kind = "person"
            elif s.kind not in ("conj", "comma", "plural"):
                last_kind = s.kind

    def _mention_for(self, p: _Parse, k: int, s: Span, group: int, ctx: Context) -> bool:
        if s.kind == "relation":
            if s.value == "self":
                slots = self._self_slot(p)
                p.mentions.append(Mention(k, slots, "self", group=group))
                return True
            prev = _prev_meaningful(p.spans, k)
            if self._is_host(p, k):
                s.used = True
                slots = [len(p.slots)]
                p.slots.append(Slot(relation=s.value, host=True))
                p.mentions.append(Mention(k, slots, s.value, group=-1))
                return False
            counted = prev is not None and prev.kind == "number" and 2 <= int(prev.value) <= 8
            if s.text in self.lex.plural_relations or counted:  # "kids", "2 friend"
                n = None
                if prev is not None and prev.kind == "number" and 1 <= int(prev.value) <= 8:
                    n = int(prev.value)
                    prev.used = True
                slots = self._new_slots(p, s.value, n or 1, plural=True)
                p.mentions.append(Mention(k, slots, s.value, plural_pending=n is None, group=group))
                return True
            if prev is not None and prev.kind == "number" and prev.value == "1":
                prev.used = True  # "one child" is a count, not an age
            slots = self._reuse_or_new(p, s.value)
            p.mentions.append(Mention(k, slots, s.value, group=group))
            return True
        if s.kind in ("we", "solo", "two_of_us", "with_others"):
            slots = self._self_slot(p)
            p.mentions.append(Mention(k, slots, "self", group=group))
            if s.kind == "with_others":  # "with my family": the speaker goes, others unnamed
                p.others_coming = True
            elif s.kind == "solo":
                p.count = 1
            elif s.kind == "two_of_us":
                p.count = 2
            return True
        if s.kind in ("one_of", "other_of"):
            return self._single_out(p, k, group)
        if s.kind == "parents":
            slots = self._reuse_or_new(p, "mother") + self._reuse_or_new(p, "father")
            p.mentions.append(Mention(k, slots, "parents", group=group))
            return True
        if s.kind == "couple":
            slots = self._self_slot(p) + self._reuse_or_new(p, "spouse")
            p.mentions.append(Mention(k, slots, "spouse", group=group))
            p.count = 2
            return True
        if s.kind == "pronoun":
            for m in reversed(p.mentions):
                if m.relation != "self" and m.group != -1:
                    p.mentions.append(Mention(k, m.slots, m.relation, group=group))
                    return True
            # Nobody named in this message: "she is retired" means someone already listed,
            # never "you". With several candidates it's a guess, so not confident.
            others = self._pronoun_candidates(s, ctx)
            if others:
                ref = others[0]
                slot = next((j for j, sl in enumerate(p.slots) if sl.ref == ref), None)
                if slot is None:
                    p.slots.append(Slot(ref=ref))
                    slot = len(p.slots) - 1
                relation = ctx.form.travellers[ref].relation
                p.mentions.append(Mention(k, [slot], relation, group=group))
                p.ambiguous += len(others) > 1
            else:
                p.mentions.append(Mention(k, [], "unknown", group=group))  # soaks up details
                p.ambiguous += 1
            return True
        return False

    @staticmethod
    def _pronoun_candidates(s: Span, ctx: Context) -> list[int]:
        """Who 'she'/'he' can mean: people in the current question first, else the latest."""
        travellers = ctx.form.travellers
        pool = [i for i, t in enumerate(travellers) if t.relation != "self"]
        asked = ctx.asked
        if asked is not None and asked.field in ("profession", "age"):
            preferred = [i for i in asked.travellers if i in pool]
            pool = preferred or pool[::-1]
        else:
            pool = pool[::-1]
        ruled_out = PRONOUN_EXCLUDES.get(s.tokens[0], frozenset())
        return [i for i in pool if travellers[i].relation not in ruled_out] or pool

    def _speaker_goes_with(self, p: _Parse) -> None:
        """'going to Dubai with 2 friends' / '2 dost ke saath': the speaker travels too."""
        people = [m for m in p.mentions if m.group != -1 and m.pos >= 0]
        if not people or any(p.slots[i].relation == "self" for m in people for i in m.slots):
            return
        first = people[0].pos
        before = _prev_meaningful(p.spans, first)
        if before is not None and before.kind == "number":  # "with 2 friends"
            before = _prev_meaningful(p.spans, first - 1)
        after = p.spans[first + 1] if first + 1 < len(p.spans) else None
        led_by_with = before is not None and before.tokens[-1] in WITH_BEFORE
        followed_by_saath = after is not None and after.tokens[-1] in WITH_AFTER
        if led_by_with or followed_by_saath:
            self._self_slot(p)

    def _single_out(self, p: _Parse, k: int, group: int) -> bool:
        """'2 friends, one of them is a lawyer, the other a student': pick one person each."""
        for m in reversed(p.mentions):
            if len(m.slots) > 1:
                free = [i for i in m.slots if i not in p.singled]
                if free:
                    p.singled.add(free[0])
                    p.mentions.append(Mention(k, [free[0]], m.relation, group=group))
                    return True
        p.mentions.append(Mention(k, [], "unknown", group=group))  # nobody to single out
        p.ambiguous += 1
        return True

    def _reuse_or_new(self, p: _Parse, relation: str) -> list[int]:
        """A second mention of the same singular relation is the same person."""
        for i, slot in enumerate(p.slots):
            if slot.relation == relation and not slot.from_plural and not slot.host:
                return [i]
        return self._new_slots(p, relation, 1, plural=False)

    def _is_host(self, p: _Parse, k: int) -> bool:
        """'visit my son' / 'bete se milne': the relation is who they visit, not a traveller."""
        prev = _prev_meaningful(p.spans, k)
        if (
            prev is not None
            and prev.value == "family_visit"
            and prev.tokens[-1] in HOST_PHRASE_ENDINGS
        ):
            return True
        nxt = p.spans[k + 1] if k + 1 < len(p.spans) else None
        return nxt is not None and nxt.value == "family_visit" and nxt.tokens[0] == "milne"

    # ------------------------------------------------------------------ attaching

    def _mentions_in(self, p: _Parse, sentence: int) -> list[Mention]:
        return [
            m
            for m in p.mentions
            if m.group != -1 and m.pos >= 0 and p.spans[m.pos].sentence == sentence
        ]

    def _nearest(self, p: _Parse, k: int, *, prefer_right: bool = False) -> Mention | None:
        same = self._mentions_in(p, p.spans[k].sentence)
        left = [m for m in same if m.pos < k]
        # A pronoun in a later clause refers to someone else: in "salaried, she is a
        # housewife" the job is not hers. "freelancer, travelling alone" still links up.
        right = [
            m
            for m in same
            if m.pos > k
            and (p.spans[m.pos].segment == p.spans[k].segment or p.spans[m.pos].kind != "pronoun")
        ]
        if prefer_right and right:
            return right[0]
        if left:
            return left[-1]
        return right[0] if right else None

    def _numbers(self, p: _Parse, ctx: Context) -> None:
        spans = p.spans
        last_age_mention: Mention | None = None
        for k, s in enumerate(spans):
            if s.kind != "number" or s.used:
                if s.kind not in ("number", "conj", "comma", "age_word"):
                    last_age_mention = None
                continue
            value = int(s.value)
            if self._negated(p, k):  # "10 not 8": the negated value is just ignored
                s.used = True
                continue
            prev = _prev_meaningful(spans, k)
            prev2 = _prev_meaningful(spans, k - 1) if prev is not None else None
            nxt = spans[k + 1] if k + 1 < len(spans) else None
            s.used = True

            if (nxt is not None and nxt.kind == "count_noun") or (
                prev is not None and prev.kind == "count_of"
            ):
                p.count = value
                if nxt is not None and nxt.tokens == ("adults",):
                    p.adults = value
                continue
            unit_next = nxt is not None and nxt.kind in ("age_unit", "month_unit")
            if (
                not unit_next
                and prev is not None
                and (
                    prev.kind == "we"
                    or (prev.kind == "plural" and prev2 is not None and prev2.kind == "we")
                )
            ):
                p.count = value
                continue

            years: int | None = None
            target: Mention | None = None
            if nxt is not None and unit_next:
                years = value if nxt.kind == "age_unit" else value // 12
                after = spans[k + 2] if k + 2 < len(spans) else None
                if after is not None and after.kind == "relation":
                    target = self._mention_at(p, k + 2)
                else:
                    target = self._nearest(p, k)
            elif prev is not None and prev.kind in ("relation", "pronoun", "we"):
                years, target = value, self._mention_at(p, k - 1)
            elif prev is not None and prev.kind == "age_word":
                years, target = value, self._nearest(p, k)
            elif prev is not None and prev.kind in ("conj", "comma") and last_age_mention:
                years, target = value, last_age_mention
            elif (kids := self._plural_before_comma(p, k)) is not None:
                years, target = value, kids  # "me and my kids, 5 and 8"
            if years is None:
                if self._group_size(p, ctx, value):
                    continue
                years = value
                if self._context_ref(p, ctx, "age", years):
                    continue
                p.ambiguous += 1
                continue
            if years > 110:
                p.ambiguous += 1
                continue
            if target is None:
                if not self._context_ref(p, ctx, "age", years):
                    p.ambiguous += 1
                continue
            self._assign_age(p, target, years)
            last_age_mention = target

    def _plural_before_comma(self, p: _Parse, k: int) -> Mention | None:
        """The uncounted 'kids' in 'my kids, 5 and 8', whose ages follow a comma."""
        j = k - 1
        while j >= 0 and p.spans[j].kind == "filler":
            j -= 1
        if j < 1 or p.spans[j].kind != "comma" or p.spans[j - 1].kind != "relation":
            return None
        m = self._mention_at(p, j - 1)
        return m if m is not None and m.plural_pending else None

    def _group_size(self, p: _Parse, ctx: Context, value: int) -> bool:
        """'3' after 'How many of your friends are travelling?' adds three friends."""
        group = ctx.form.pending_group
        asked = ctx.asked is not None and ctx.asked.field == "group_size"
        named = any(m.relation != "self" for m in p.mentions)
        if not asked or group is None or named or not 1 <= value <= MAX_TRAVELLERS:
            return False
        self._new_slots(p, group, value, plural=True)
        return True

    def _uncounted_groups(self, p: _Parse, ctx: Context) -> None:
        """'with my friends' / 'the kids': people are coming, but not how many.

        One bare slot would silently become one person, so it is reported as
        ``group_of`` instead and the bot asks for the number. Ages or jobs given
        ("kids 5 and 8", "friends are students") keep the people as they are.
        """
        if p.count is not None:
            return  # "6 of us, me and my friends": the count already says how many
        known = {t.relation for t in ctx.form.travellers}
        for m in p.mentions:
            if not m.plural_pending or len(m.slots) != 1 or m.relation in known:
                continue
            slot = p.slots[m.slots[0]]
            details = (slot.ref, slot.age, slot.profession, slot.country, slot.purpose)
            if slot.remove or any(v is not None for v in details):
                continue
            slot.group = True
            p.group_of = m.relation

    @staticmethod
    def _trip_details(p: _Parse) -> int:
        people = any(not sl.host for sl in p.slots)
        return sum(bool(x) for x in (p.country, p.purpose, p.other_country, people))

    @staticmethod
    def _has_unknown(p: _Parse) -> bool:
        """'how does my phone compare to other phones': unknown words around a lone
        profession word mean it isn't someone saying their job."""
        return any(s.kind == "unknown" for s in p.spans)

    @staticmethod
    def _negated(p: _Parse, k: int) -> bool:
        """'not tourism', 'no 8': the value right after a negation is being ruled out."""
        prev = _prev_meaningful(p.spans, k)
        return prev is not None and (prev.kind == "deny" or prev.tokens == ("not",))

    def _mention_at(self, p: _Parse, pos: int) -> Mention | None:
        for m in p.mentions:
            if m.pos == pos:
                return m if m.group != -1 else None
        return None

    def _assign_age(self, p: _Parse, m: Mention, years: int) -> None:
        free = [i for i in m.slots if p.slots[i].age is None]
        if free:
            p.slots[free[0]].age = years
        elif m.plural_pending and len(p.slots) < MAX_COUNT:
            new = self._new_slots(p, m.relation, 1, plural=True)
            m.slots += new
            p.slots[new[0]].age = years
        else:
            p.ambiguous += 1

    def _professions(self, p: _Parse, ctx: Context) -> None:
        for k, s in enumerate(p.spans):
            if s.kind == "business_word":
                asked_job = ctx.asked is not None and ctx.asked.field == "profession"
                if asked_job or self._mentions_in(p, s.sentence):
                    s.kind, s.value = "profession", "business_owner"
                else:
                    s.kind, s.value = "purpose", "business"
                    continue
            if s.kind != "profession" or self._negated(p, k):
                continue
            # "newborn son", "retired couple": a profession right before a person is theirs.
            nxt = p.spans[k + 1] if k + 1 < len(p.spans) else None
            ahead = (
                self._mention_at(p, k + 1) if nxt is not None and nxt.kind in PERSON_KINDS else None
            )
            target = ahead or self._nearest(p, k)
            if target is None:
                if not self._context_ref(p, ctx, "profession", s.value):
                    p.ambiguous += 1
                continue
            for m in self._spread(p, target, k, s):
                for i in m.slots:
                    slot = p.slots[i]
                    if slot.profession is None or s.value == NO_PROFESSION_SWAP:
                        slot.profession = s.value

    def _spread(self, p: _Parse, target: Mention, k: int, s: Span) -> list[Mention]:
        """'me and my wife are doctors' -> the profession applies to the whole group."""
        group = [m for m in p.mentions if m.group == target.group and m.pos < k]
        if len(group) < 2:
            return [target]
        between = p.spans[group[-1].pos + 1 : k]
        plural_word = s.text.endswith("s") and self.lex.lookup(tuple(s.text[:-1].split())) == (
            "profession",
            s.value,
        )
        has_we = any(p.spans[m.pos].kind == "we" for m in group)
        if plural_word or has_we or any(b.kind == "plural" for b in between):
            return group
        return [target]

    def _context_ref(self, p: _Parse, ctx: Context, what: str, value: str | int) -> bool:
        """Attach a detail with no person in the message to whoever the bot asked about."""
        asked = ctx.asked
        targets: tuple[int, ...] = ()
        if asked is not None and asked.field == what and asked.travellers:
            taken = sum(1 for sl in p.slots if sl.ref is not None and getattr(sl, what) is not None)
            # "both retired" / "dono retired". With two jobs ("business, both are
            # students") they are answered in order instead of all getting the first.
            single = what == "age" or self._single_job(p)
            if taken == 0 and single and any(s.kind == "plural" for s in p.spans):
                targets = asked.travellers
            elif taken < len(asked.travellers):
                targets = (asked.travellers[taken],)
        if not targets and what == "profession":
            # A bare profession with one person still missing one is theirs. Bare numbers
            # never get this fallback: "what is 2+2" must not become someone's age.
            lacking = [i for i, t in enumerate(ctx.form.travellers) if t.profession is None]
            if len(lacking) == 1:
                targets = (lacking[0],)
            elif not ctx.form.travellers and not p.mentions and not self._has_unknown(p):
                me = self._self_slot(p)[0]
                p.mentions.append(Mention(-1, [me], "self"))
                p.slots[me].profession = str(value)
                return True
        if not targets:
            return False
        for ref in targets:
            slot = next((sl for sl in p.slots if sl.ref == ref), None)
            if slot is None:
                slot = Slot(ref=ref)
                p.slots.append(slot)
            if what == "age":
                slot.age = int(value)
            else:
                slot.profession = str(value)
        return True

    @staticmethod
    def _single_job(p: _Parse) -> bool:
        return sum(s.kind in ("profession", "business_word") for s in p.spans) == 1

    def _places_and_purposes(self, p: _Parse, ctx: Context) -> None:
        trip = {"country": ctx.form.country, "purpose": ctx.form.purpose}
        found: dict[str, str | None] = {"country": None, "purpose": None}
        for k, s in enumerate(p.spans):
            if s.kind == "other_country" and not self._negated(p, k):
                p.other_country = p.other_country or s.value
                continue
            if s.kind not in ("country", "purpose") or self._negated(p, k):
                continue
            kind = s.kind
            current = found[kind] or trip[kind]
            if current is None:
                found[kind] = s.value
                continue
            if s.value == current:
                continue
            people = [
                m
                for m in p.mentions
                if m.group != -1
                and m.relation != "self"
                and m.pos >= 0
                and p.spans[m.pos].segment == s.segment
            ]
            if people:
                for m in people:
                    for i in m.slots:
                        setattr(p.slots[i], kind, s.value)
            elif found[kind] is not None and trip[kind] is None:
                if kind == "purpose":  # "tourism and a conference": ask which is the main one
                    for v in (found[kind], s.value):
                        if v is not None and v not in p.purpose_choices:
                            p.purpose_choices.append(v)
                else:
                    p.ambiguous += 1  # two different countries for the whole trip
            else:
                found[kind] = s.value
                p.conflict = True
        p.country, p.purpose = found["country"], found["purpose"]
        if len(p.purpose_choices) > 1:
            p.purpose = None

    def _removals(self, p: _Parse) -> None:
        for k, s in enumerate(p.spans):
            if s.kind == "remove":
                target = self._nearest(p, k)
                if target is None or target.relation == "self":
                    p.ambiguous += 1
                    continue
                for i in target.slots:
                    p.slots[i].remove = True

    # ------------------------------------------------------------------ result

    def _build(self, p: _Parse, ctx: Context, message: str) -> Extraction:
        kinds = {s.kind for s in p.spans}
        if "restart" in kinds:
            return Extraction(intent="restart")
        if "off_topic" in kinds and self._trip_details(p) < 2:
            # "translate this to French", "capital of France": one place name next to an
            # off-topic word is not a trip. "ignore that. Japan tourism, just me" still is.
            return Extraction(on_topic=False)

        patches = tuple(
            TravellerPatch(
                ref=sl.ref,
                relation=sl.relation,
                profession=sl.profession,
                age=sl.age,
                country=sl.country,
                purpose=sl.purpose,
                remove=sl.remove,
            )
            for sl in p.slots
            if not sl.host and not sl.group
        )
        bare_me = patches == (TravellerPatch(relation="self"),)
        asking = bool(kinds & {"help", "office_question"})
        if (
            bare_me
            and not (p.country or p.purpose or p.count or p.others_coming or p.group_of)
            and (p.unexplained or asking)
        ):
            patches = ()  # "write me a poem": 'me' is not a traveller here
        if p.adults is not None:
            kids = sum(1 for sl in p.slots if sl.relation in self.lex.ask_age_for and not sl.host)
            p.count = p.adults + kids
        asked_who = ctx.asked is not None and ctx.asked.field == "travellers"
        if "no_more" in kinds or (asked_who and kinds & {"deny"} and not p.slots):
            # "that's all" / "no" to "who else?": everyone known so far is the whole group
            # (a "me" in the same message is not one more person when you're already listed)
            has_self = any(t.relation == "self" for t in ctx.form.travellers)
            new_people = sum(
                1 for x in patches if x.ref is None and not (has_self and x.relation == "self")
            )
            p.count = len(ctx.form.travellers) + new_people or None
        details = bool(
            p.country or p.purpose or p.count or patches or p.purpose_choices or p.group_of
        )
        intent: Intent
        if details:
            corrective = p.conflict or kinds & {"correct_cue", "deny", "remove"}
            intent = "correct" if corrective else "give_details"
        elif "deny" in kinds:
            intent = "correct"
        elif "confirm" in kinds:
            intent = "confirm"
        else:
            intent = "give_details"

        signal = kinds & {"confirm", "deny", "greeting", "help", "office_question", "no_more"}
        words = [s for s in p.spans if s.kind not in ("sep", "comma")]
        on_topic = (
            details
            or bool(signal)
            or p.other_country is not None
            or has_devanagari(message)
            or not words  # "???": unsure, not off-topic
            or p.unexplained < 2  # one odd word ("umm") is "didn't catch that"
        )
        if len(patches) > MAX_TRAVELLERS:  # keep the first 8; the count tells the bot to say so
            p.count = max(p.count or 0, len(patches))
            patches = patches[:MAX_TRAVELLERS]
        count = p.count if p.count is not None and 1 <= p.count <= MAX_COUNT else None
        if p.count is not None and count is None:
            p.ambiguous += 1
        return Extraction(
            on_topic=on_topic,
            intent=intent,
            country=p.country,
            purpose=p.purpose,
            traveller_count=count,
            travellers=patches,
            other_country=p.other_country,
            others_coming=p.others_coming,
            group_of=p.group_of,
            purpose_choices=tuple(p.purpose_choices[:3]),
            asks_office="office_question" in kinds,
            asks_help="help" in kinds,
        )


def _prev_meaningful(spans: list[Span], k: int) -> Span | None:
    """The span before ``k`` that is not a filler."""
    for j in range(k - 1, -1, -1):
        if spans[j].kind != "filler":
            return spans[j]
    return None
