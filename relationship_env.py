"""Relationship Simulator: shared module for the RL tool-calling workshop.

This file is used by the training notebook (`rl_toolcall_relationship.ipynb`),
the Streamlit demo (`app.py`), and the tests. It is pure Python (stdlib only),
so it runs on a CPU without torch. Only `tool_schemas()` needs `transformers`,
and it imports it lazily on first use.

The game in one sentence: the model reads how the partner feels, thinks inside
`<think>...</think>`, and calls wholesome tools (gifts, movies, dinner, hugs, ...)
that a deterministic simulator scores by how much they raise the "love meter".
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import itertools
import json
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, NamedTuple, Optional, Sequence, Union

# =============================================================================
# 1. Catalog
# =============================================================================
# The Catalog is the fixed "menu" of things the partner tools accept. Because
# the menu is finite, the simulator can try every possible plan and find the
# best one, which gives us an objective ground truth for rewards and benchmarks.

#: Each gift item maps to one gift type. Preferences talk about gift *types*
#: ("she loves flowers"), tools take gift *items* ("roses").
GIFT_ITEMS: dict[str, str] = {
    "roses": "flowers",
    "sunflowers": "flowers",
    "dark_chocolate": "chocolate",
    "truffles": "chocolate",
    "novel": "book",
    "poetry_book": "book",
    "necklace": "jewelry",
    "teddy_bear": "plush_toy",
    "headphones": "gadget",
    "scented_candle": "home",
}

#: All gift types, sorted so the order is stable across runs.
GIFT_TYPES: list[str] = sorted(set(GIFT_ITEMS.values()))

#: Movie genres for `watch_movie(genre)`.
GENRES: list[str] = ["action", "animation", "comedy", "documentary", "horror", "romance"]

#: Dishes for `cook_dinner(dish)`.
DISHES: list[str] = ["curry", "pancakes", "pasta", "soup", "sushi", "tacos"]

#: Date activities for `plan_date(activity)`.
ACTIVITIES: list[str] = ["beach_walk", "cooking_class", "museum_visit", "picnic", "stargazing"]

# =============================================================================
# 2. Partner state vocabulary
# =============================================================================

#: How the partner feels right now.
MOODS: list[str] = ["angry", "happy", "sad", "stressed", "tired"]

#: The situation the scenario takes place in.
OCCASIONS: list[str] = [
    "after_argument",
    "anniversary",
    "bad_day_at_work",
    "birthday",
    "normal_evening",
    "rainy_weekend",
]

#: The five love languages. An action that matches the partner's love
#: language earns a small bonus in the simulator.
LOVE_LANGUAGES: list[str] = [
    "acts_of_service",
    "gifts",
    "physical_touch",
    "quality_time",
    "words_of_affirmation",
]

#: One-sentence hint per occasion, returned by `ask_how_partner_feels`.
OCCASION_HINTS: dict[str, str] = {
    "after_argument": "You two had a disagreement earlier, and a sincere apology would mean a lot.",
    "anniversary": "Today is your anniversary, so a thoughtful plan together would feel special.",
    "bad_day_at_work": "Work was rough today, and some comfort and attention would help.",
    "birthday": "It's their birthday, so a little celebration would be lovely.",
    "normal_evening": "It's a calm, ordinary evening, perfect for a small kind gesture.",
    "rainy_weekend": "It's a rainy weekend, a nice time for something cozy indoors.",
}

# =============================================================================
# 3. Free-text arguments and reward scale
# =============================================================================
# Most tool arguments must come from the Catalog. Two arguments are free text:
# the apology reason and the message text. The simulator ignores their content
# (only "non-empty and at most FREE_TEXT_MAX characters" matters), so a model
# cannot game the score by writing ever-sweeter messages.

#: (tool name, argument name) pairs whose value is free text.
FREE_TEXT_ARGS: frozenset[tuple[str, str]] = frozenset(
    {("apologize", "reason"), ("send_message", "text")}
)

#: Maximum length of a free-text argument, in characters.
FREE_TEXT_MAX: int = 200

#: Placeholder free text used during optimal-plan search and in examples.
CANONICAL_TEXT: str = "I love you"

#: Love_Delta that maps to an outcome reward of 1.0 (reward = delta / scale, clipped).
OUTCOME_SCALE: int = 30


# =============================================================================
# 4. Data models
# =============================================================================
# All models are frozen (immutable) dataclasses: a scenario cannot change by
# accident while it is being scored, and equal objects compare equal.


def _canonical_json(value: Any) -> str:
    """Serialize `value` to one stable JSON string (sorted keys, no extra spaces)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=repr)


@dataclass(frozen=True)
class Preferences:
    """What the partner likes. Each field draws from the Catalog lists above."""

    favorite_gift_types: tuple[str, ...]  # 1-2 of GIFT_TYPES
    liked_genres: tuple[str, ...]  # 1-2 of GENRES
    disliked_genres: tuple[str, ...]  # 1 of GENRES, disjoint from liked_genres
    favorite_food: str  # one of DISHES
    love_language: str  # one of LOVE_LANGUAGES

    def to_dict(self) -> dict:
        return {
            "favorite_gift_types": list(self.favorite_gift_types),
            "liked_genres": list(self.liked_genres),
            "disliked_genres": list(self.disliked_genres),
            "favorite_food": self.favorite_food,
            "love_language": self.love_language,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Preferences":
        return cls(
            favorite_gift_types=tuple(d["favorite_gift_types"]),
            liked_genres=tuple(d["liked_genres"]),
            disliked_genres=tuple(d["disliked_genres"]),
            favorite_food=d["favorite_food"],
            love_language=d["love_language"],
        )


@dataclass(frozen=True)
class PartnerState:
    """The partner at the start of a scenario: mood, love meter, occasion, preferences."""

    mood: str  # one of MOODS
    love_meter: int  # 0..100 (scenarios start in 10..65)
    occasion: str  # one of OCCASIONS
    preferences: Preferences

    def to_dict(self) -> dict:
        return {
            "mood": self.mood,
            "love_meter": self.love_meter,
            "occasion": self.occasion,
            "preferences": self.preferences.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PartnerState":
        return cls(
            mood=d["mood"],
            love_meter=int(d["love_meter"]),
            occasion=d["occasion"],
            preferences=Preferences.from_dict(d["preferences"]),
        )


@dataclass(frozen=True, eq=False)
class ToolCall:
    """One tool call, e.g. `ToolCall("give_gift", {"item": "roses"})`.

    `arguments` is a plain dict (the shape models emit), but equality and
    hashing use its canonical JSON, so two calls with the same values are equal
    regardless of key order and a ToolCall can live in a set.
    """

    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        # Copy so later changes to the caller's dict cannot alter this call.
        object.__setattr__(self, "arguments", dict(self.arguments or {}))

    def canonical(self) -> str:
        """Canonical JSON of the full call (name + all arguments)."""
        return _canonical_json({"name": self.name, "arguments": self.arguments})

    def key(self) -> tuple[str, str]:
        """(name, canonical JSON of Catalog args) with free-text args removed.

        The simulator uses this for repeat detection: `send_message("hi")` and
        `send_message("hello")` count as the same action.
        """
        catalog_args = {
            k: v for k, v in self.arguments.items() if (self.name, k) not in FREE_TEXT_ARGS
        }
        return (self.name, _canonical_json(catalog_args))

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": dict(self.arguments)}

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCall":
        return cls(name=d["name"], arguments=dict(d.get("arguments") or {}))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return NotImplemented
        return self.canonical() == other.canonical()

    def __hash__(self) -> int:
        return hash(self.canonical())

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.name}({args})"


@dataclass(frozen=True)
class Scenario:
    """One Relationship Simulator episode.

    A tool scenario has an `optimal_plan`: a tuple of Call_Groups, where the
    calls inside one group may run in any order. A No_Tool_Scenario instead has
    `no_tool_keywords` that a correct plain-text answer must contain.
    """

    scenario_id: str  # sha1 key of state + request
    partner_name: str
    pronoun: Literal["she", "he"]
    state: PartnerState
    request: str
    optimal_plan: Optional[tuple[tuple[ToolCall, ...], ...]] = None  # None for No_Tool
    optimal_delta: Optional[int] = None
    no_tool_keywords: Optional[tuple[str, ...]] = None

    @property
    def is_no_tool(self) -> bool:
        """True when the correct behavior is a plain-text answer with no tool call."""
        return self.optimal_plan is None

    def to_dict(self) -> dict:
        """JSON-ready dictionary (tuples become lists)."""
        return {
            "scenario_id": self.scenario_id,
            "partner_name": self.partner_name,
            "pronoun": self.pronoun,
            "state": self.state.to_dict(),
            "request": self.request,
            "optimal_plan": None
            if self.optimal_plan is None
            else [[call.to_dict() for call in group] for group in self.optimal_plan],
            "optimal_delta": self.optimal_delta,
            "no_tool_keywords": None
            if self.no_tool_keywords is None
            else list(self.no_tool_keywords),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        """Inverse of `to_dict`: `Scenario.from_dict(s.to_dict()) == s`."""
        plan = d.get("optimal_plan")
        keywords = d.get("no_tool_keywords")
        delta = d.get("optimal_delta")
        return cls(
            scenario_id=d["scenario_id"],
            partner_name=d["partner_name"],
            pronoun=d["pronoun"],
            state=PartnerState.from_dict(d["state"]),
            request=d["request"],
            optimal_plan=None
            if plan is None
            else tuple(tuple(ToolCall.from_dict(c) for c in group) for group in plan),
            optimal_delta=None if delta is None else int(delta),
            no_tool_keywords=None if keywords is None else tuple(keywords),
        )

    def to_json(self) -> str:
        """Scenario as a JSON string (stored in the dataset `scenario` column)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Scenario":
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class RewardBreakdown:
    """The four parts of the Tool_Reward. `total` lies in [-1.0, 3.0].

    - format:  0.5 for a leading, non-empty <think> block
    - syntax:  0.5 for well-formed tool calls with known tool names
    - outcome: Love_Delta / OUTCOME_SCALE, clipped to [-1, 1]
    - exact:   1.0 when the calls match the Optimal_Plan
    """

    format: float
    syntax: float
    outcome: float
    exact: float

    @property
    def total(self) -> float:
        return self.format + self.syntax + self.outcome + self.exact


# =============================================================================
# 5. Partner_Tools
# =============================================================================
# These nine functions are what the model "sees". Their type hints and
# Google-style docstrings are turned into JSON schemas by
# `transformers.utils.get_json_schema`, and the chat template shows those
# schemas to the model.
#
# The functions are schema carriers only: the real effect of a call is computed
# by the Relationship_Sim (section 6), which keeps the partner state explicit.
#
# Arguments that accept only Catalog values end their docstring line with
# `(choices: [...])`. `get_json_schema` turns that into an `enum`, so the model
# is told exactly which values are allowed. The choice lists are filled in from
# the Catalog below, so the schemas can never drift from the Catalog.

_SCHEMA_ONLY = {"error": "Partner_Tools are schema carriers; run calls with RelationshipSim.execute."}


def _choices(values: Iterable[str]) -> str:
    """Render a Catalog list as `(choices: ["a", "b"])` (JSON, as get_json_schema expects)."""
    return f"(choices: {json.dumps(sorted(values))})"


def give_gift(item: str) -> dict:
    """
    Give the partner a small gift.

    Args:
        item: The gift to give. {choices}
    """
    return dict(_SCHEMA_ONLY)


def watch_movie(genre: str) -> dict:
    """
    Watch a movie together on the couch.

    Args:
        genre: The movie genre to watch. {choices}
    """
    return dict(_SCHEMA_ONLY)


def cook_dinner(dish: str) -> dict:
    """
    Cook dinner for the partner.

    Args:
        dish: The dish to cook. {choices}
    """
    return dict(_SCHEMA_ONLY)


def hug() -> dict:
    """
    Give the partner a warm, comforting hug.
    """
    return dict(_SCHEMA_ONLY)


def kiss() -> dict:
    """
    Give the partner a light, affectionate kiss on the cheek.
    """
    return dict(_SCHEMA_ONLY)


def apologize(reason: str) -> dict:
    """
    Sincerely apologize to the partner.

    Args:
        reason: A short reason for the apology (non-empty, at most 200 characters).
    """
    return dict(_SCHEMA_ONLY)


def plan_date(activity: str) -> dict:
    """
    Plan a date together.

    Args:
        activity: The date activity to plan. {choices}
    """
    return dict(_SCHEMA_ONLY)


def send_message(text: str) -> dict:
    """
    Send the partner a kind text message.

    Args:
        text: The message to send (non-empty, at most 200 characters).
    """
    return dict(_SCHEMA_ONLY)


def ask_how_partner_feels() -> dict:
    """
    Ask the partner how they feel and listen to the answer.
    """
    return dict(_SCHEMA_ONLY)


# Fill the `{choices}` placeholders from the Catalog (runs once at import time).
give_gift.__doc__ = give_gift.__doc__.format(choices=_choices(GIFT_ITEMS))
watch_movie.__doc__ = watch_movie.__doc__.format(choices=_choices(GENRES))
cook_dinner.__doc__ = cook_dinner.__doc__.format(choices=_choices(DISHES))
plan_date.__doc__ = plan_date.__doc__.format(choices=_choices(ACTIVITIES))

#: All nine tools, in a fixed order (also the order of `tool_schemas()`).
PARTNER_TOOLS: list[Callable[..., dict]] = [
    give_gift,
    watch_movie,
    cook_dinner,
    hug,
    kiss,
    apologize,
    plan_date,
    send_message,
    ask_how_partner_feels,
]

#: Tool name -> function, handy for lookups such as "is this a known tool?".
TOOLS_BY_NAME: dict[str, Callable[..., dict]] = {fn.__name__: fn for fn in PARTNER_TOOLS}

# Schemas are computed once and reused (building them parses every docstring).
_SCHEMA_CACHE: dict[str, dict] = {}


def tool_schemas(names: Optional[Iterable[str]] = None) -> list[dict]:
    """Return the JSON schema of each Partner_Tool, as passed to `apply_chat_template(tools=...)`.

    Args:
        names: Tool names to include, in the order given. `None` means all nine
            tools in `PARTNER_TOOLS` order.

    Raises:
        KeyError: If a name is not one of the Partner_Tools.
    """
    if not _SCHEMA_CACHE:
        # Imported here so the rest of the module works without transformers.
        from transformers.utils import get_json_schema

        for fn in PARTNER_TOOLS:
            _SCHEMA_CACHE[fn.__name__] = get_json_schema(fn)

    selected = list(TOOLS_BY_NAME) if names is None else list(names)
    unknown = [n for n in selected if n not in _SCHEMA_CACHE]
    if unknown:
        raise KeyError(f"Unknown Partner_Tool name(s): {unknown}")
    # Deep copies, so a caller editing a schema cannot corrupt the cache.
    return [copy.deepcopy(_SCHEMA_CACHE[n]) for n in selected]


# =============================================================================
# 6. Relationship_Sim
# =============================================================================
# The simulator scores tool calls with a small, fixed rules table. Everything
# is deterministic: the same partner state and the same calls always give the
# same love meter, which is what makes "the best plan" well defined.
#
# Each call is scored in this order (the first matching step wins):
#
#   step 0  unknown tool / bad argument -> not executed, returns {"error": ...}
#                                          (still uses up one call of the turn)
#   step 1  4th or later call this turn -> -5   (keep plans short)
#   step 2  same tool + same Catalog args already executed -> 0
#                                          (free text is ignored, so two
#                                          different messages count as a repeat)
#   step 3  after_argument and no apology yet, tool is not apologize -> -10
#   step 4  otherwise: BASE_RULES value + LOVE_LANGUAGE_BONUS if it matches
#
# After each call the love meter is clamped to 0..100, and the call's
# love_delta is the change the meter actually made.

#: Arguments of each tool (all required). Argument-less tools map to ().
TOOL_ARGS: dict[str, tuple[str, ...]] = {
    "give_gift": ("item",),
    "watch_movie": ("genre",),
    "cook_dinner": ("dish",),
    "hug": (),
    "kiss": (),
    "apologize": ("reason",),
    "plan_date": ("activity",),
    "send_message": ("text",),
    "ask_how_partner_feels": (),
}

#: Allowed values of each Catalog argument: (tool, argument) -> sorted values.
CATALOG_ARGS: dict[tuple[str, str], list[str]] = {
    ("give_gift", "item"): sorted(GIFT_ITEMS),
    ("watch_movie", "genre"): sorted(GENRES),
    ("cook_dinner", "dish"): sorted(DISHES),
    ("plan_date", "activity"): sorted(ACTIVITIES),
}

#: Penalties and limits from the table above.
MAX_CALLS_PER_TURN: int = 3  # calls after this many in one turn get OVER_LIMIT_VALUE
OVER_LIMIT_VALUE: int = -5  # step 1
REPEAT_VALUE: int = 0  # step 2
UNAPOLOGIZED_VALUE: int = -10  # step 3: after_argument before any apology
LOVE_LANGUAGE_BONUS: int = 5  # step 4 bonus

#: Which tools earn the love-language bonus for each love language.
LOVE_LANGUAGE_TOOLS: dict[str, frozenset[str]] = {
    "gifts": frozenset({"give_gift"}),
    "quality_time": frozenset({"plan_date", "watch_movie"}),
    "physical_touch": frozenset({"hug", "kiss"}),
    "words_of_affirmation": frozenset({"send_message"}),
    "acts_of_service": frozenset({"cook_dinner"}),
}

#: Moods in which a hug helps the most.
COMFORT_MOODS: frozenset[str] = frozenset({"sad", "tired", "stressed"})

#: Base rule of each tool: (partner state, arguments) -> raw love value.
#: Read each line as "this tool is worth X when ..., otherwise Y".
BASE_RULES: dict[str, Callable[[PartnerState, dict], int]] = {
    # +20 for a gift of a favorite type, +5 for any other gift
    "give_gift": lambda s, a: 20 if GIFT_ITEMS[a["item"]] in s.preferences.favorite_gift_types else 5,
    # +10 liked genre, -15 disliked genre, 0 otherwise
    "watch_movie": lambda s, a: 10
    if a["genre"] in s.preferences.liked_genres
    else (-15 if a["genre"] in s.preferences.disliked_genres else 0),
    # +12 for the favorite food, +4 for any other dish
    "cook_dinner": lambda s, a: 12 if a["dish"] == s.preferences.favorite_food else 4,
    # +8 when sad / tired / stressed, +3 otherwise
    "hug": lambda s, a: 8 if s.mood in COMFORT_MOODS else 3,
    # +5 when happy, -5 when angry, +2 otherwise
    "kiss": lambda s, a: 5 if s.mood == "happy" else (-5 if s.mood == "angry" else 2),
    # +15 after an argument, -2 otherwise (apologizing for nothing is odd)
    "apologize": lambda s, a: 15 if s.occasion == "after_argument" else -2,
    # +10 on a birthday or anniversary, +5 otherwise
    "plan_date": lambda s, a: 10 if s.occasion in ("birthday", "anniversary") else 5,
    # +6 always (the text itself is not scored)
    "send_message": lambda s, a: 6,
    # +5 on a bad day at work, +1 otherwise
    "ask_how_partner_feels": lambda s, a: 5 if s.occasion == "bad_day_at_work" else 1,
}


def _clamp(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, value))


def _readable(value: str) -> str:
    """`dark_chocolate` -> `dark chocolate` for result texts."""
    return value.replace("_", " ")


def validate_call(call: ToolCall) -> Optional[str]:
    """Return an error message if `call` cannot be executed, else `None`.

    Checks: known tool name, no missing or unexpected arguments, Catalog values
    from the Catalog, and free text that is non-empty and at most
    FREE_TEXT_MAX characters (after stripping whitespace).
    """
    if call.name not in TOOL_ARGS:
        return f"Unknown tool '{call.name}'. Available tools: {', '.join(TOOL_ARGS)}."
    expected = TOOL_ARGS[call.name]
    for arg in call.arguments:
        if arg not in expected:
            allowed = ", ".join(expected) if expected else "none"
            return f"Unexpected argument '{arg}' for {call.name}. Allowed arguments: {allowed}."
    for arg in expected:
        if arg not in call.arguments:
            return f"Missing required argument '{arg}' for {call.name}."
        value = call.arguments[arg]
        if (call.name, arg) in CATALOG_ARGS:
            allowed_values = CATALOG_ARGS[(call.name, arg)]
            if not isinstance(value, str) or value not in allowed_values:
                return (
                    f"Invalid value {value!r} for argument '{arg}' of {call.name}. "
                    f"Allowed values: {', '.join(allowed_values)}."
                )
        elif (call.name, arg) in FREE_TEXT_ARGS:
            text = value.strip() if isinstance(value, str) else ""
            if not text or len(text) > FREE_TEXT_MAX:
                return (
                    f"Argument '{arg}' of {call.name} must be non-empty text of at most "
                    f"{FREE_TEXT_MAX} characters."
                )
    return None


def _reaction(name: str, delta: int) -> str:
    """A short, wholesome reaction sentence that matches the size of the change."""
    name = name[:1].upper() + name[1:]  # sentence start ("your partner" -> "Your partner")
    if delta >= 10:
        return f"{name} lights up with a big smile."
    if delta > 0:
        return f"{name} smiles warmly."
    if delta == 0:
        return f"{name} doesn't react much."
    return f"{name} doesn't seem pleased."


def _action_text(name: str, call: ToolCall) -> str:
    """What happened, in one sentence (the reaction is appended separately)."""
    a = call.arguments
    texts = {
        "give_gift": lambda: f"You give {name} the {_readable(a.get('item', ''))}.",
        "watch_movie": lambda: f"You watch a {_readable(a.get('genre', ''))} movie together.",
        "cook_dinner": lambda: f"You cook {_readable(a.get('dish', ''))} for {name}.",
        "hug": lambda: f"You give {name} a warm hug.",
        "kiss": lambda: f"You give {name} a light kiss on the cheek.",
        "apologize": lambda: f"You sincerely apologize to {name}.",
        "plan_date": lambda: f"You plan a {_readable(a.get('activity', ''))} date with {name}.",
        "send_message": lambda: f"You send {name} a kind message.",
    }
    return texts[call.name]()


class Episode:
    """Mutable execution state for one scenario.

    Used by the Tool_Reward (replaying a completion's calls) and the Demo_App
    (one chat session). `state` is the current partner state: it starts as the
    scenario's state and its `love_meter` changes as calls run.
    """

    def __init__(self, scenario: Union[Scenario, PartnerState], partner_name: Optional[str] = None):
        if isinstance(scenario, Scenario):
            state = scenario.state
            name = partner_name or scenario.partner_name
        else:
            state = scenario
            name = partner_name or "your partner"
        self.partner_name: str = name
        self.state: PartnerState = state  # frozen, replaced on every meter change
        self.initial_meter: int = state.love_meter
        self.executed: list[ToolCall] = []  # successfully executed calls, in order
        self.apologized: bool = False
        self.turn_call_count: int = 0  # calls in the current user turn, incl. invalid ones

    @property
    def love_meter(self) -> int:
        return self.state.love_meter

    @property
    def love_delta(self) -> int:
        """Total change of the love meter since the episode started."""
        return self.state.love_meter - self.initial_meter

    def begin_turn(self) -> None:
        """Start a new user turn: resets the per-turn call count (history is kept)."""
        self.turn_call_count = 0

    def _raw_value(self, call: ToolCall) -> int:
        """Steps 1-4 of the rules table for a call that passed validation."""
        if self.turn_call_count > MAX_CALLS_PER_TURN:  # step 1 (count already includes this call)
            return OVER_LIMIT_VALUE
        if any(prev.key() == call.key() for prev in self.executed):  # step 2
            return REPEAT_VALUE
        if self.state.occasion == "after_argument" and not self.apologized and call.name != "apologize":
            return UNAPOLOGIZED_VALUE  # step 3
        value = BASE_RULES[call.name](self.state, call.arguments)  # step 4
        if call.name in LOVE_LANGUAGE_TOOLS.get(self.state.preferences.love_language, frozenset()):
            value += LOVE_LANGUAGE_BONUS
        return value

    def execute(self, call: ToolCall) -> dict:
        """Run one call and return `{"result", "love_delta", "love_meter"}` or `{"error"}`.

        An invalid call changes nothing except the turn's call count.
        """
        self.turn_call_count += 1
        error = validate_call(call)
        if error is not None:
            return {"error": error}

        raw = self._raw_value(call)
        before = self.state.love_meter
        after = _clamp(before + raw)
        self.state = dataclasses.replace(self.state, love_meter=after)
        self.executed.append(call)
        if call.name == "apologize":
            self.apologized = True  # also on repeats and over-limit calls
        delta = after - before

        if call.name == "ask_how_partner_feels":
            name = self.partner_name[:1].upper() + self.partner_name[1:]
            result = f"{name} feels {self.state.mood}. {OCCASION_HINTS[self.state.occasion]}"
        else:
            result = f"{_action_text(self.partner_name, call)} {_reaction(self.partner_name, delta)}"
        return {"result": result, "love_delta": delta, "love_meter": after}


# -----------------------------------------------------------------------------
# Optimal_Plan search
# -----------------------------------------------------------------------------
# The Catalog is small, so we can simply try every plan:
#   10 gifts + 6 genres + 6 dishes + 5 activities + 5 other tools = 32 actions
#   32 + 32**2 + 32**3 = 33,824 plans of 1 to 3 calls.
# Free-text tools use CANONICAL_TEXT (the text is never scored anyway).


def action_space(tools: Optional[Iterable[str]] = None) -> list[ToolCall]:
    """Every distinct action of the given tools (default: all nine), in tie-break order."""
    selected = TOOL_ARGS if tools is None else tools
    actions = []
    for name in selected:
        args = TOOL_ARGS[name]
        if not args:  # hug(), kiss(), ask_how_partner_feels()
            actions.append(ToolCall(name, {}))
        elif (name, args[0]) in FREE_TEXT_ARGS:  # apologize(reason), send_message(text)
            actions.append(ToolCall(name, {args[0]: CANONICAL_TEXT}))
        else:  # one call per Catalog value
            for value in CATALOG_ARGS[(name, args[0])]:
                actions.append(ToolCall(name, {args[0]: value}))
    return sorted(actions, key=_tie_break_key)


def _tie_break_key(call: ToolCall) -> tuple[str, str]:
    """Lexicographic tie-break key of one call: (name, sorted-JSON args)."""
    return (call.name, json.dumps(call.arguments, sort_keys=True))


class _SearchCall(ToolCall):
    """A ToolCall that computes `key()` once. Search actions never change, and
    the repeat check calls `key()` ~130k times per search, so this is a big speedup."""

    def key(self) -> tuple[str, str]:
        cached = self.__dict__.get("_cached_key")
        if cached is None:
            cached = super().key()
            object.__setattr__(self, "_cached_key", cached)  # dataclass is frozen
        return cached


def _clone(episode: Episode) -> Episode:
    """Cheap copy of an Episode, so one prefix can be extended by many actions."""
    twin = copy.copy(episode)  # state is frozen, counters are ints: safe to share
    twin.executed = list(episode.executed)
    return twin


@functools.lru_cache(maxsize=4096)
def _cached_optimal_plan(
    state: PartnerState, tools: tuple[str, ...], max_calls: int
) -> tuple[tuple[ToolCall, ...], ...]:
    """Exhaustive search + grouping. Cached by (state, tools, max_calls); all are hashable."""
    # `actions` is sorted by tie-break key, so comparing tuples of action
    # indices is the same as comparing the [(name, args), ...] lists.
    actions = [_SearchCall(a.name, a.arguments) for a in action_space(tools)]
    best_rank: Optional[tuple] = None
    best_plan: tuple[ToolCall, ...] = ()

    # Depth-first search. Each prefix is executed once and then cloned for
    # every next action, which gives the same Love_Delta as replaying the full
    # sequence (the simulator is deterministic) but is much faster.
    def extend(episode: Episode, plan: tuple[int, ...]) -> None:
        nonlocal best_rank, best_plan
        for i, action in enumerate(actions):
            child = _clone(episode)
            child.execute(action)
            seq = plan + (i,)
            # Smaller rank is better: higher delta, then fewer calls, then lexicographic.
            rank = (-child.love_delta, len(seq), seq)
            if best_rank is None or rank < best_rank:
                best_rank, best_plan = rank, tuple(actions[j] for j in seq)
            if len(seq) < max_calls:
                extend(child, seq)

    extend(Episode(state), ())
    return _group_calls(state, best_plan)


def _group_calls(state: PartnerState, plan: Sequence[ToolCall]) -> tuple[tuple[ToolCall, ...], ...]:
    """Split a plan into Call_Groups.

    A call joins the current group only if *every* order of the group's calls
    (after the earlier groups) gives the same Love_Delta. Checking all
    permutations, not just one swap, guarantees any in-group order is optimal.
    """
    groups: list[list[ToolCall]] = []
    prefix: list[ToolCall] = []  # calls of the finished groups
    current: list[ToolCall] = []
    for call in plan:
        candidate = current + [call]
        if current and _order_free(state, prefix, candidate):
            current = candidate
        else:
            if current:
                groups.append(current)
                prefix += current
            current = [call]
    if current:
        groups.append(current)
    return tuple(tuple(g) for g in groups)


def _order_free(state: PartnerState, prefix: list[ToolCall], group: list[ToolCall]) -> bool:
    """True if every order of `group`, run after `prefix`, gives the same Love_Delta."""
    deltas = {_love_delta(state, prefix + list(order)) for order in itertools.permutations(group)}
    return len(deltas) == 1


def _love_delta(state: PartnerState, calls: Sequence[ToolCall]) -> int:
    """Love_Delta of running `calls` as one turn from `state`."""
    episode = Episode(state)
    for call in calls:
        episode.execute(call)
    return episode.love_delta


class RelationshipSim:
    """Deterministic Relationship Simulator: episodes, scoring, plans, scenarios."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def new_episode(self, scenario: Union[Scenario, PartnerState]) -> Episode:
        """A fresh Episode for `scenario` (or a bare PartnerState)."""
        return Episode(scenario)

    def love_delta(self, state: Union[PartnerState, Scenario], calls: Sequence[ToolCall]) -> int:
        """Love_Delta of running `calls` as one turn from `state` (final - initial meter)."""
        episode = Episode(state)
        for call in calls:
            episode.execute(call)
        return episode.love_delta

    def optimal_plan(
        self,
        state: Union[PartnerState, Scenario],
        allowed_tools: Optional[Sequence[str]] = None,
        max_calls: int = MAX_CALLS_PER_TURN,
    ) -> list[list[ToolCall]]:
        """The best plan of 1 to `max_calls` calls, as a list of Call_Groups.

        The search tries every sequence of actions (see `action_space`) and
        keeps the one with the highest Love_Delta. Ties go to fewer calls, then
        to the lexicographically smallest `[(name, json.dumps(args, sort_keys=True)), ...]`.
        The plan is then split into Call_Groups: calls inside one group can be
        emitted in any order without changing the Love_Delta.

        Args:
            state: The partner state (or a Scenario, whose state is used).
            allowed_tools: Restrict the search to these tool names. `None` means
                all nine Partner_Tools.
            max_calls: Longest plan to consider (1 to 3).

        Raises:
            KeyError: If `allowed_tools` contains an unknown tool name.
            ValueError: If `max_calls` is not in 1..3 or `allowed_tools` is empty.
        """
        if isinstance(state, Scenario):
            state = state.state
        if allowed_tools is None:
            tools = tuple(TOOL_ARGS)
        else:
            unknown = [t for t in allowed_tools if t not in TOOL_ARGS]
            if unknown:
                raise KeyError(f"Unknown Partner_Tool name(s): {unknown}")
            tools = tuple(sorted(set(allowed_tools)))
            if not tools:
                raise ValueError("allowed_tools must contain at least one tool")
        if not 1 <= max_calls <= MAX_CALLS_PER_TURN:
            raise ValueError(f"max_calls must be between 1 and {MAX_CALLS_PER_TURN}")

        groups = _cached_optimal_plan(state, tools, max_calls)
        # Fresh ToolCall objects, so callers cannot edit the cached plan.
        return [[ToolCall(c.name, c.arguments) for c in group] for group in groups]

    def generate_scenarios(
        self,
        n: int,
        split: Literal["train", "eval"],
        exclude: Iterable[str] = frozenset(),
    ) -> list[Scenario]:
        """Generate `n` seeded Scenarios for one split.

        - Each split has its own random stream, `Random(f"{seed}-{split}")`, so
          the same seed always gives the same list.
        - Occasions are assigned round-robin (every occasion appears at least
          `n // 6` times); moods cycle with a per-occasion offset, so every
          mood shows up for every occasion that has 5 or more scenarios.
        - Exactly `round(0.10 * n)` scenarios are No_Tool_Scenarios: a question
          about the partner's preferences, answered in plain text.
        - Tool scenarios are resampled until their best plan reaches
          `optimal_delta >= OUTCOME_SCALE`.
        - `scenario_id` is the sha1 of the state JSON + request; ids in
          `exclude` (pass the train ids when generating eval) are skipped, and
          ids never repeat within one list.

        Raises:
            ValueError: If `n` is negative.
            RuntimeError: If 100 attempts in a row fail to produce a usable scenario.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        rng = random.Random(f"{self.seed}-{split}")
        seen: set[str] = set(exclude)
        n_no_tool = round(0.10 * n)
        no_tool_slots = set(rng.sample(range(n), n_no_tool)) if n else set()
        per_occasion = {occ: 0 for occ in OCCASIONS}

        scenarios = []
        for i in range(n):
            occasion = OCCASIONS[i % len(OCCASIONS)]
            k = per_occasion[occasion]
            per_occasion[occasion] += 1
            # Offset by occasion index so every occasion starts on a different mood.
            mood = MOODS[(k + OCCASIONS.index(occasion)) % len(MOODS)]
            make = self._make_no_tool if i in no_tool_slots else self._make_tool
            for _attempt in range(MAX_SCENARIO_ATTEMPTS):
                scenario = make(rng, mood, occasion)
                if scenario is not None and scenario.scenario_id not in seen:
                    break
            else:
                raise RuntimeError(
                    f"Could not generate a usable {occasion}/{mood} scenario "
                    f"after {MAX_SCENARIO_ATTEMPTS} attempts"
                )
            seen.add(scenario.scenario_id)
            scenarios.append(scenario)
        return scenarios

    # -- generation helpers ---------------------------------------------------

    @staticmethod
    def _draw_partner(rng: random.Random) -> tuple[str, str]:
        pronoun = rng.choice(("she", "he"))
        return rng.choice(PARTNER_NAMES[pronoun]), pronoun

    @staticmethod
    def _draw_state(rng: random.Random, mood: str, occasion: str) -> PartnerState:
        gift_types = tuple(sorted(rng.sample(GIFT_TYPES, rng.randint(1, 2))))
        genres = rng.sample(GENRES, rng.randint(1, 2) + 1)  # liked + one disliked, disjoint
        prefs = Preferences(
            favorite_gift_types=gift_types,
            liked_genres=tuple(sorted(genres[:-1])),
            disliked_genres=(genres[-1],),
            favorite_food=rng.choice(DISHES),
            love_language=rng.choice(LOVE_LANGUAGES),
        )
        return PartnerState(mood, rng.randint(METER_MIN, METER_MAX), occasion, prefs)

    def _make_tool(self, rng: random.Random, mood: str, occasion: str) -> Optional[Scenario]:
        """One tool scenario, or None if its best plan is below OUTCOME_SCALE."""
        name, pronoun = self._draw_partner(rng)
        state = self._draw_state(rng, mood, occasion)
        request = _fill(rng.choice(REQUEST_TEMPLATES[occasion]), name, pronoun)
        groups = _cached_optimal_plan(state, tuple(TOOL_ARGS), MAX_CALLS_PER_TURN)
        delta = _love_delta(state, [c for g in groups for c in g])
        if delta < OUTCOME_SCALE:
            return None
        plan = tuple(tuple(ToolCall(c.name, c.arguments) for c in g) for g in groups)
        return Scenario(
            scenario_id=scenario_key(state, request),
            partner_name=name,
            pronoun=pronoun,
            state=state,
            request=request,
            optimal_plan=plan,
            optimal_delta=delta,
        )

    def _make_no_tool(self, rng: random.Random, mood: str, occasion: str) -> Scenario:
        """One No_Tool_Scenario: a preference question with keyword answers."""
        name, pronoun = self._draw_partner(rng)
        state = self._draw_state(rng, mood, occasion)
        topic = rng.choice(sorted(NO_TOOL_TEMPLATES))
        request = _fill(rng.choice(NO_TOOL_TEMPLATES[topic]), name, pronoun)
        values = getattr(state.preferences, topic)
        values = (values,) if isinstance(values, str) else values
        return Scenario(
            scenario_id=scenario_key(state, request),
            partner_name=name,
            pronoun=pronoun,
            state=state,
            request=request,
            no_tool_keywords=tuple(normalize_keyword(v) for v in values),
        )


# -----------------------------------------------------------------------------
# Scenario generation vocabulary
# -----------------------------------------------------------------------------

#: Attempts per scenario slot before the generator gives up.
MAX_SCENARIO_ATTEMPTS: int = 100

#: Generated scenarios start with a love meter in this range (inclusive).
METER_MIN: int = 10
METER_MAX: int = 65

#: Fixed partner names per pronoun.
PARTNER_NAMES: dict[str, list[str]] = {
    "she": ["Mia", "Emma", "Sofia", "Hana", "Lina", "Aisha", "Grace", "Priya"],
    "he": ["Noah", "Liam", "Kenji", "Omar", "Lucas", "Ethan", "Ravi", "Daniel"],
}

#: Pronoun forms used by the templates.
PRONOUN_FORMS: dict[str, dict[str, str]] = {
    "she": {"pronoun": "she", "Pronoun": "She", "object": "her", "possessive": "her"},
    "he": {"pronoun": "he", "Pronoun": "He", "object": "him", "possessive": "his"},
}

#: User requests per occasion. Placeholders: {name}, {pronoun}, {Pronoun}, {object}, {possessive}.
REQUEST_TEMPLATES: dict[str, list[str]] = {
    "after_argument": [
        "We just had an argument and {pronoun}'s upset. What should I do?",
        "{name} and I had a disagreement earlier. How can I make things right?",
        "I said something thoughtless to {name} today. Can you help me make it up to {object}?",
    ],
    "anniversary": [
        "It's our anniversary today. Help me make it special for {name}.",
        "Today is my anniversary with {name}. What would {pronoun} love?",
        "I want to celebrate our anniversary with {name}. Any ideas?",
    ],
    "bad_day_at_work": [
        "{name} just got home after a rough day at work. How can I help?",
        "Work was really hard for {name} today. What can I do to cheer {object} up?",
        "{name} had a long, tough day at the office. How can I comfort {object}?",
    ],
    "birthday": [
        "It's {name}'s birthday today. Help me make {object} smile.",
        "Today is {name}'s birthday. What should I do for {object}?",
        "I want to make {name}'s birthday one to remember. Any ideas?",
    ],
    "normal_evening": [
        "It's a quiet evening at home with {name}. How can I make {possessive} day nicer?",
        "Just a regular evening with {name}. What's a sweet thing I could do?",
        "{name} and I are relaxing tonight. How can I show {object} I care?",
    ],
    "rainy_weekend": [
        "It's raining all weekend and {name} is stuck inside with me. What should we do?",
        "Rainy weekend at home with {name}. How can I make it cozy for {object}?",
        "The weather is gloomy this weekend. How can I brighten {name}'s day?",
    ],
}

#: No_Tool questions per Preferences field. The answer keywords are that field's values.
NO_TOOL_TEMPLATES: dict[str, list[str]] = {
    "favorite_food": [
        "Quick question: what's {name}'s favorite food?",
        "Remind me, what dish does {name} like most?",
    ],
    "love_language": [
        "What is {name}'s love language?",
        "How does {name} most like to receive love? Just tell me {possessive} love language.",
    ],
    "liked_genres": [
        "Which movie genres does {name} enjoy?",
        "What kinds of movies does {name} like to watch?",
    ],
    "disliked_genres": [
        "Which movie genre does {name} not like?",
        "Is there a kind of movie {name} would rather avoid?",
    ],
    "favorite_gift_types": [
        "What kind of gifts does {name} like best?",
        "Remind me which types of gifts make {name} happiest.",
    ],
}


def _fill(template: str, name: str, pronoun: str) -> str:
    return template.format(name=name, **PRONOUN_FORMS[pronoun])


def normalize_keyword(value: str) -> str:
    """No_Tool keyword form: lowercase, underscores as spaces (`plush_toy` -> `plush toy`)."""
    return " ".join(value.lower().replace("_", " ").split())


def scenario_key(state: PartnerState, request: str) -> str:
    """sha1 of the canonical state JSON + request text (the Scenario id)."""
    return hashlib.sha1((_canonical_json(state.to_dict()) + request).encode("utf-8")).hexdigest()


# =============================================================================
# 7. Tool_Call_Parser and Tool_Call_Printer
# =============================================================================
#
# Qwen-style tool-call text looks like this:
#
#   <think>She is upset, so apologize first.</think>
#   <tool_call>
#   {"name": "apologize", "arguments": {"reason": "I'm sorry."}}
#   </tool_call>
#
# `parse_tool_calls` turns raw model output into a ParseResult; `print_tool_calls`
# does the reverse. Rewards, the benchmark, and the demo app all use these two
# functions, so they all read model output the same way.

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class ParseError:
    """A problem with one Tool_Call_Block.

    `index` is the zero-based block index (-1 for an internal parser error).
    `reason` is one of: "invalid JSON: ...", "body not an object", "missing name",
    "arguments not an object", "missing </tool_call>", "internal: ...".
    """

    index: int
    reason: str


@dataclass
class ParseResult:
    think: Optional[str]        # text inside the first <think>...</think>, or None
    calls: list[ToolCall]       # well-formed tool calls, in output order
    text: str                   # remaining plain text (no blocks, no think), stripped
    errors: list[ParseError]    # one entry per malformed block
    think_leading: bool         # output starts with exactly one <think>...</think>


def _parse_block_body(index: int, body: str) -> Union[ToolCall, ParseError]:
    """Decode one block body into a ToolCall, or explain why it is malformed."""
    try:
        data = json.loads(body.strip())
    except (ValueError, RecursionError) as exc:
        return ParseError(index, f"invalid JSON: {exc}")
    if not isinstance(data, dict):
        return ParseError(index, "body not an object")
    name = data.get("name")
    if not isinstance(name, str):
        return ParseError(index, "missing name")
    arguments = data.get("arguments", {})  # missing `arguments` means no arguments
    if isinstance(arguments, str):
        # Some models emit arguments as a JSON string: "{\"item\": \"roses\"}".
        try:
            arguments = json.loads(arguments)
        except (ValueError, RecursionError):
            return ParseError(index, "arguments not an object")
    if not isinstance(arguments, dict):
        return ParseError(index, "arguments not an object")
    return ToolCall(name, arguments)


def _parse_tool_calls_unguarded(output: str) -> ParseResult:
    calls: list[ToolCall] = []
    errors: list[ParseError] = []
    outside: list[str] = []  # text segments that are not inside any block

    # Step 1: scan for blocks. Each <tool_call> opener starts one block.
    pos = 0
    index = 0
    while True:
        start = output.find(TOOL_CALL_OPEN, pos)
        if start == -1:
            outside.append(output[pos:])
            break
        outside.append(output[pos:start])
        body_start = start + len(TOOL_CALL_OPEN)
        end = output.find(TOOL_CALL_CLOSE, body_start)
        if end == -1:
            # No closer: this block swallows the rest of the output.
            errors.append(ParseError(index, "missing </tool_call>"))
            break
        parsed = _parse_block_body(index, output[body_start:end])
        if isinstance(parsed, ParseError):
            errors.append(parsed)
        else:
            calls.append(parsed)
        pos = end + len(TOOL_CALL_CLOSE)
        index += 1

    # Step 2: think extraction, on text outside the blocks only.
    outside_text = "\n".join(outside)
    think: Optional[str] = None
    think_start = outside_text.find(THINK_OPEN)
    think_end = -1
    if think_start != -1:
        think_end = outside_text.find(THINK_CLOSE, think_start + len(THINK_OPEN))
    if think_start != -1 and think_end != -1:
        think = outside_text[think_start + len(THINK_OPEN):think_end]
        remaining = outside_text[:think_start] + outside_text[think_end + len(THINK_CLOSE):]
    else:
        remaining = outside_text

    think_leading = (
        think is not None
        and output.strip().startswith(THINK_OPEN)
        and outside_text.count(THINK_OPEN) == 1
    )

    return ParseResult(
        think=think,
        calls=calls,
        text=remaining.strip(),
        errors=errors,
        think_leading=think_leading,
    )


def parse_tool_calls(output: str) -> ParseResult:
    """Split raw model output into think text, tool calls, plain text, and errors.

    Never raises: any unexpected problem becomes a ParseError with index -1.
    """
    try:
        if not isinstance(output, str):
            output = str(output)
        return _parse_tool_calls_unguarded(output)
    except Exception as exc:  # robustness: callers must always get a result
        return ParseResult(
            think=None,
            calls=[],
            text="",
            errors=[ParseError(-1, f"internal: {type(exc).__name__}: {exc}")],
            think_leading=False,
        )


def _dumps_call(call: ToolCall) -> str:
    """JSON for one call. `</` becomes `<\\/` (still valid JSON) so a string
    argument containing `</tool_call>` cannot close the block early."""
    text = json.dumps({"name": call.name, "arguments": call.arguments}, ensure_ascii=False)
    return text.replace("</", "<\\/")


def print_tool_calls(calls: Sequence[ToolCall]) -> str:
    """Render calls as Qwen Tool_Call_Blocks, one per call, joined by newlines."""
    return "\n".join(
        f"{TOOL_CALL_OPEN}\n{_dumps_call(call)}\n{TOOL_CALL_CLOSE}" for call in calls
    )


# =============================================================================
# 8. Tool_Reward
# =============================================================================
#
# The reward in plain words (total between -1.0 and 3.0):
#
#   format  +0.5  the answer starts with one non-empty <think>...</think> block
#   syntax  +0.5  every <tool_call> block is valid JSON and names a known tool
#   outcome -1..1 how much the calls raised the love meter, divided by 30
#                 (so +30 points or more gives the full +1.0)
#   exact   +1.0  the calls are exactly the best plan the simulator found
#
# For a question that needs no tool ("What is her favorite food?") the outcome
# is +1.0 and exact is +1.0 when the plain-text answer names every keyword,
# and the outcome is -1.0 when the model calls a tool anyway.
#
# The content of <think> is never scored, only whether it is there. That way
# RL shapes *what the model does*, not *what it writes while thinking*.

_LOG = logging.getLogger("relationship_env")
_SIM = RelationshipSim()  # stateless scorer; love_delta never uses the seed


def _free_text_ok(value: Any) -> bool:
    """Free-text argument check: a string of 1..FREE_TEXT_MAX chars (after strip)."""
    return isinstance(value, str) and 0 < len(value.strip()) <= FREE_TEXT_MAX


def plan_matches(calls: Sequence[ToolCall], optimal_plan: Sequence[Sequence[ToolCall]]) -> bool:
    """True when `calls` follow the Optimal_Plan.

    The flat call list is cut into consecutive chunks with the same sizes as
    the plan's Call_Groups. Each chunk must contain the same calls as its group
    in any order (compared by name + Catalog arguments). Free-text arguments
    (apology reason, message text) only need to be non-empty and short enough.
    """
    sizes = [len(group) for group in optimal_plan]
    if len(calls) != sum(sizes):
        return False
    for call in calls:
        for arg in TOOL_ARGS.get(call.name, ()):
            if (call.name, arg) in FREE_TEXT_ARGS and not _free_text_ok(call.arguments.get(arg)):
                return False
    start = 0
    for group, size in zip(optimal_plan, sizes):
        chunk = calls[start:start + size]
        start += size
        if Counter(c.key() for c in chunk) != Counter(c.key() for c in group):
            return False
    return True


def _has_all_keywords(text: str, keywords: Iterable[str]) -> bool:
    """Normalized substring test: every keyword appears in the answer text."""
    haystack = normalize_keyword(text)
    return all(normalize_keyword(k) in haystack for k in keywords)


def score_completion(
    text: str,
    scenario: Scenario,
    tool_names: Optional[Iterable[str]] = None,
    scale: float = OUTCOME_SCALE,
) -> RewardBreakdown:
    """Score one completion against one Scenario (see the table above).

    `tool_names` are the tools offered in the prompt (default: all nine).
    Only well-formed calls count toward the outcome; malformed blocks just
    lose the syntax points.
    """
    names = set(TOOL_ARGS) if tool_names is None else set(tool_names)
    parse = parse_tool_calls(text)

    fmt = 0.5 if parse.think_leading and parse.think and parse.think.strip() else 0.0
    syntax = 0.5 if not parse.errors and all(c.name in names for c in parse.calls) else 0.0

    if scenario.is_no_tool:
        if parse.calls:
            outcome, exact = -1.0, 0.0
        elif _has_all_keywords(parse.text, scenario.no_tool_keywords or ()):
            outcome, exact = 1.0, 1.0
        else:
            outcome, exact = 0.0, 0.0
    else:
        delta = _SIM.love_delta(scenario.state, parse.calls)
        outcome = max(-1.0, min(1.0, delta / scale))
        exact = 1.0 if plan_matches(parse.calls, scenario.optimal_plan) else 0.0

    return RewardBreakdown(format=fmt, syntax=syntax, outcome=float(outcome), exact=exact)


@functools.lru_cache(maxsize=8192)
def _decode_scenario(text: str) -> Scenario:
    """Scenario from its JSON column value (cached: each prompt has many completions)."""
    return Scenario.from_json(text)


def _as_scenario(value: Any) -> Scenario:
    if isinstance(value, Scenario):
        return value
    if isinstance(value, dict):
        return Scenario.from_dict(value)
    return _decode_scenario(value)


def _schema_name(schema: Any) -> Optional[str]:
    """Tool name from a JSON schema ({"type": "function", "function": {...}} or flat) or a callable."""
    if callable(schema):
        return getattr(schema, "__name__", None)
    if isinstance(schema, dict):
        inner = schema.get("function", schema)
        name = inner.get("name") if isinstance(inner, dict) else None
        return name if isinstance(name, str) else None
    return None


@functools.lru_cache(maxsize=1024)
def _names_from_json(text: str) -> frozenset[str]:
    return _names_from_schemas(json.loads(text))


def _names_from_schemas(schemas: Any) -> frozenset[str]:
    if isinstance(schemas, str):
        return _names_from_json(schemas)
    if schemas is None:
        return frozenset(TOOL_ARGS)
    return frozenset(n for n in (_schema_name(s) for s in schemas) if n)


def _tool_names_per_row(tools: Any, n: int) -> list[Any]:
    """Normalize the `tools` argument to one entry per completion.

    Accepted: None (all nine tools), one JSON string, one list of schemas, or a
    per-row column (list of JSON strings / schema lists) as TRL passes it.
    """
    if tools is None or isinstance(tools, str):
        return [tools] * n
    tools = list(tools)
    per_row = len(tools) == n and all(t is None or isinstance(t, (str, list, tuple)) for t in tools)
    return tools if per_row else [tools] * n


def _completion_text(completion: Any) -> str:
    """Plain text of a completion: a string, or a conversational message list."""
    if isinstance(completion, list):
        return completion[0]["content"] if completion else ""
    if isinstance(completion, dict):
        return completion["content"]
    return str(completion)


def tool_reward(completions, scenario, tools=None, **kwargs) -> list[float]:
    """TRL reward function for GRPOTrainer / RLOOTrainer.

    `completions`: strings or conversational message lists.
    `scenario`: the dataset column of Scenario JSON strings, one per completion.
    Returns one float per completion, in order. A completion that cannot be
    scored gets 0.0; the first error of the batch is logged once.
    """
    completions = list(completions)
    n = len(completions)
    scenarios = list(scenario) if isinstance(scenario, (list, tuple)) else [scenario] * n
    tool_rows = _tool_names_per_row(tools, n)

    rewards: list[float] = []
    parts: list[RewardBreakdown] = []
    first_error: Optional[str] = None
    for i, completion in enumerate(completions):
        try:
            names = _names_from_schemas(tool_rows[i])
            breakdown = score_completion(_completion_text(completion), _as_scenario(scenarios[i]), names)
            rewards.append(float(breakdown.total))
            parts.append(breakdown)
        except Exception as exc:  # never crash the trainer on one bad sample
            rewards.append(0.0)
            if first_error is None:
                first_error = f"completion {i}: {type(exc).__name__}: {exc}"
    if first_error is not None:
        _LOG.warning("tool_reward: scored 0.0 for failed completions; first error: %s", first_error)

    log_metric = kwargs.get("log_metric")
    if callable(log_metric) and parts:
        try:
            for field in ("format", "syntax", "outcome", "exact"):
                log_metric(f"reward/{field}", sum(getattr(p, field) for p in parts) / len(parts))
        except Exception:  # metric logging is optional
            pass
    return rewards


# =============================================================================
# 9. AST_Matcher (BFCL-style)
# =============================================================================
#
# How the Berkeley Function Calling Leaderboard (BFCL) "AST" check works, in
# plain words: we never run the predicted calls. We only compare their shape
# (the abstract syntax tree: function name + argument names + values) with a
# ground-truth answer key.
#
#   * the function name must be the expected one,
#   * every argument the schema marks as required must be there,
#   * no argument may appear that the schema does not know,
#   * each given argument must equal one of the accepted answers
#     ("5" vs 5.0? 5 == 5.0 is fine, True == 1 is not; strings ignore case and
#     outer spaces), and free-text arguments just need 1..200 characters.
#
# Categories: `simple` / `multiple` expect exactly one call; `parallel` expects
# several calls in any order (same count, each GT call paired with one
# prediction); `irrelevance` is correct only when the model calls nothing.

BFCL_CATEGORIES: tuple[str, ...] = ("simple", "multiple", "parallel", "irrelevance")


@dataclass(frozen=True)
class FreeText:
    """Marker for a Free_Text_Argument: any string of 1..max_len chars after strip."""

    max_len: int = FREE_TEXT_MAX

    def accepts(self, value: Any) -> bool:
        return isinstance(value, str) and 0 < len(value.strip()) <= self.max_len


@dataclass(frozen=True)
class GTCall:
    """One ground-truth call: name + acceptable values per argument.

    `args[arg]` is a list of acceptable values, or `FreeText()` (the class
    `FreeText` itself is accepted too). Arguments not listed are unchecked
    (they only need to exist in the schema).
    """

    name: str
    args: dict = dataclasses.field(default_factory=dict)


def values_equal(a: Any, b: Any) -> bool:
    """Type-aware equality used by the AST_Matcher.

    bool only equals bool; int/float equal when numerically equal; strings
    compared after `strip().lower()`; lists/tuples elementwise; dicts key-wise.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k]) for k in a)
    if a is None or b is None:
        return a is None and b is None
    return type(a) is type(b) and a == b


def _schema_index(schemas: Optional[Iterable[Any]]) -> dict[str, dict]:
    """name -> parameters dict, for OpenAI-style or bare {"name", "parameters"} schemas."""
    index: dict[str, dict] = {}
    for schema in schemas or ():
        if not isinstance(schema, dict):
            continue
        inner = schema.get("function", schema)
        if isinstance(inner, dict) and isinstance(inner.get("name"), str):
            params = inner.get("parameters")
            index[inner["name"]] = params if isinstance(params, dict) else {}
    return index


def _accepts(expected: Any, value: Any) -> bool:
    """True if `value` is one of the acceptable GT values (or valid free text)."""
    if expected is FreeText:
        expected = FreeText()
    if isinstance(expected, FreeText):
        return expected.accepts(value)
    options = expected if isinstance(expected, (list, tuple)) else [expected]
    return any(values_equal(value, option) for option in options)


def _call_matches(pred: ToolCall, gt: GTCall, schemas: dict[str, dict]) -> bool:
    """Name, required args, no unknown args, and acceptable values (see above)."""
    if pred.name != gt.name or not isinstance(pred.arguments, dict):
        return False
    params = schemas.get(gt.name)
    if params is None:  # no schema: GT argument names are the known properties
        properties, required = set(gt.args), set()
    else:
        properties = set((params.get("properties") or {}).keys())
        required = set(params.get("required") or ())
    args = pred.arguments
    if not required <= args.keys():
        return False
    if any(name not in properties for name in args):
        return False
    return all(_accepts(gt.args[name], value) for name, value in args.items() if name in gt.args)


def _perfect_matching(match: list[list[bool]]) -> bool:
    """Backtracking search: can every GT row be paired with a distinct prediction?"""
    n = len(match)
    used = [False] * n

    def assign(row: int) -> bool:
        if row == n:
            return True
        for col in range(n):
            if match[row][col] and not used[col]:
                used[col] = True
                if assign(row + 1):
                    return True
                used[col] = False
        return False

    return assign(0)


def ast_match(pred: Sequence[ToolCall], gt: Sequence[GTCall], category: str, schemas: Iterable[Any]) -> bool:
    """BFCL-style AST check of predicted calls against the ground truth."""
    pred, gt = list(pred), list(gt)
    if category == "irrelevance":
        return not pred
    index = _schema_index(schemas)
    if category in ("simple", "multiple"):
        return len(pred) == 1 and len(gt) == 1 and _call_matches(pred[0], gt[0], index)
    if category == "parallel":
        if len(pred) != len(gt):
            return False
        return _perfect_matching([[_call_matches(p, g, index) for p in pred] for g in gt])
    raise ValueError(f"unknown BFCL category: {category!r}")


# =============================================================================
# 10. Benchmark helpers
# =============================================================================
#
# The notebook's results table has four Headline_Metrics. In plain words:
#
#   BFCL AST Acc      How often the model writes the *right* tool call (right
#                     function, right arguments) for a request. Mean of the
#                     `simple`, `multiple` and `parallel` category accuracies.
#   Irrelevance Acc   How often the model correctly calls *nothing* when no
#                     offered tool fits the request.
#   Sim Task Success  How often the model's calls are exactly the best plan the
#                     simulator found (or, for a question that needs no tool,
#                     a plain answer naming every keyword).
#   Mean Love Gain    How many love-meter points the model's calls earn on
#                     average. Broken <tool_call> blocks are ignored; only the
#                     well-formed calls are run.
#
# A metric with no examples (e.g. the public dataset failed to load and a
# category ended up empty) is stored as `None` (JSON `null`) and shown as N/A.

#: Results-table column names, in display order, keyed by the JSON field name.
HEADLINE_COLUMNS: dict[str, str] = {
    "bfcl_ast_acc": "BFCL AST Acc",
    "irrelevance_acc": "Irrelevance Acc",
    "sim_task_success": "Sim Task Success",
    "mean_love_gain": "Mean Love Gain",
}
#: Metrics shown as percentages; the rest (love gain) are love-meter points.
PERCENT_METRICS: frozenset[str] = frozenset({"bfcl_ast_acc", "irrelevance_acc", "sim_task_success"})
NA = "N/A"
MINUS = "\u2212"  # typographic minus sign "−" used in the delta table


class SimItemResult(NamedTuple):
    """Outcome of one model answer on one eval Scenario."""

    success: bool  # counts toward Sim Task Success
    love_gain: int  # Love_Delta of the well-formed calls (Mean Love Gain)


def sim_item_result(text: str, scenario: Scenario, sim: Optional[RelationshipSim] = None) -> SimItemResult:
    """Score one completion for Sim Task Success and Mean Love Gain.

    Success uses the same rules as the reward's exact-match bonus: the calls
    follow the Optimal_Plan, or for a No_Tool_Scenario there is no tool call and
    the answer contains every keyword. Love gain runs only the well-formed calls.
    """
    sim = _SIM if sim is None else sim
    parse = parse_tool_calls(text)
    if scenario.is_no_tool:
        success = not parse.calls and _has_all_keywords(parse.text, scenario.no_tool_keywords or ())
    else:
        success = plan_matches(parse.calls, scenario.optimal_plan)
    return SimItemResult(success=bool(success), love_gain=int(sim.love_delta(scenario.state, parse.calls)))


def accuracy(flags: Iterable[bool]) -> Optional[float]:
    """Fraction of True values, or None when there are no items."""
    flags = list(flags)
    return sum(1 for f in flags if f) / len(flags) if flags else None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """Mean of the non-None values, or None when nothing is available."""
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None


def aggregate_headline(
    bfcl_per_category: dict[str, Optional[float]],
    sim_success: Sequence[bool],
    love_gains: Sequence[float],
) -> dict[str, Optional[float]]:
    """The four Headline_Metrics of one model (None = N/A).

    `bfcl_per_category` maps each BFCL category to its accuracy (None if the
    category had no items). `sim_success` / `love_gains` hold one value per
    eval Scenario, e.g. from `sim_item_result`.
    """
    return {
        "bfcl_ast_acc": _mean(bfcl_per_category.get(c) for c in ("simple", "multiple", "parallel")),
        "irrelevance_acc": bfcl_per_category.get("irrelevance"),
        "sim_task_success": accuracy(sim_success),
        "mean_love_gain": _mean(love_gains),
    }


def format_pct(value: Optional[float]) -> str:
    """0.314 -> '31.4%'; None -> 'N/A'."""
    return NA if value is None else f"{value * 100:.1f}%"


def format_gain(value: Optional[float]) -> str:
    """6.25 -> '6.2' love-meter points; None -> 'N/A'."""
    return NA if value is None else f"{value:.1f}"


def format_delta(delta: Optional[float], percent: bool = False) -> str:
    """Signed change vs the base model: '+5.0%' / '−2.5' (U+2212); None -> 'N/A'.

    With `percent=True` the delta is a fraction and is shown in percentage
    points. A change that rounds to zero is shown as '+0.0'.
    """
    if delta is None:
        return NA
    shown = round(delta * 100 if percent else delta, 1)
    sign = MINUS if shown < 0 else "+"
    return f"{sign}{abs(shown):.1f}{'%' if percent else ''}"


def headline_deltas(model: dict[str, Optional[float]], base: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    """Per-metric change `model - base`; None when either side is N/A."""
    return {
        k: None if model.get(k) is None or base.get(k) is None else model[k] - base[k]
        for k in HEADLINE_COLUMNS
    }


def format_headline_row(headline: dict[str, Optional[float]]) -> dict[str, str]:
    """One results-table row: {'BFCL AST Acc': '31.4%', ..., 'Mean Love Gain': '6.2'}."""
    return {
        label: format_pct(headline.get(k)) if k in PERCENT_METRICS else format_gain(headline.get(k))
        for k, label in HEADLINE_COLUMNS.items()
    }


def format_delta_row(deltas: dict[str, Optional[float]]) -> dict[str, str]:
    """One delta-table row with signed values, e.g. {'BFCL AST Acc': '+5.0%', ...}."""
    return {label: format_delta(deltas.get(k), percent=k in PERCENT_METRICS) for k, label in HEADLINE_COLUMNS.items()}


def _jsonable(value: Any) -> Any:
    """Convert ToolCalls, Scenarios, tuples, sets and NaN into plain JSON values."""
    if isinstance(value, (ToolCall, Scenario, PartnerState)):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and value != value:  # NaN -> null
        return None
    return value


def build_results_dict(
    base_model: str,
    seed: int,
    decoding: dict,
    counts: dict[str, int],
    headline: dict[str, dict[str, Optional[float]]],
    bfcl_per_category: dict[str, dict[str, Optional[float]]],
    skipped_models: Sequence[str] = (),
    samples: Sequence[dict] = (),
) -> dict:
    """Everything the Results_File stores (unavailable metrics stay None -> null).

    `headline` / `bfcl_per_category` are keyed by model name ('base', 'sft', ...).
    `samples` holds one dict per prediction: model, category, item_id, output,
    think, calls, correct, love_delta, reward.
    """
    return _jsonable({
        "base_model": base_model,
        "seed": seed,
        "decoding": dict(decoding),
        "counts": dict(counts),
        "headline": {m: {k: h.get(k) for k in HEADLINE_COLUMNS} for m, h in headline.items()},
        "bfcl_per_category": {m: {c: per.get(c) for c in BFCL_CATEGORIES} for m, per in bfcl_per_category.items()},
        "skipped_models": list(skipped_models),
        "samples": list(samples),
    })


def write_results_file(path: Union[str, os.PathLike], results: Optional[dict] = None, **fields: Any) -> Path:
    """Write the Results_File as JSON and return its path.

    Pass a dict from `build_results_dict`, or the same keyword arguments that
    `build_results_dict` takes. Missing parent folders are created.
    """
    data = _jsonable(results) if results is not None else build_results_dict(**fields)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return out


def require_sft(path: Union[str, os.PathLike]) -> Path:
    """Return the SFT_Checkpoint path, or raise if the SFT stage has not run yet."""
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"SFT checkpoint not found at {p}. Run the SFT stage first.")
    return p


# =============================================================================
# 11. Prompt_Renderer
# =============================================================================
#
# Every prompt in the workshop (SFT, DPO, GRPO, RLOO, eval, and the app) is
# built here, so the model always sees the same text in every stage.
#
# Why "inline" the tools? SFT and DPO pass the `tools` column to the chat
# template, which writes a tool description block into the system message.
# GRPO and RLOO (TRL v1.12.0) ignore that column, so for them we render the
# tool block once and paste it into the system message ourselves. A check
# makes sure the result is byte-for-byte what the template would produce.
#
# The tokenizer is duck-typed: anything with a Hugging Face style
# `apply_chat_template(messages, tools=..., tokenize=False, ...)` works.

#: Qwen chat markers around the system block.
SYSTEM_START = "<|im_start|>system\n"
TURN_END = "<|im_end|>"

#: Readable occasion phrases for the system message.
OCCASION_PHRASES: dict[str, str] = {
    "after_argument": "right after a disagreement",
    "anniversary": "your anniversary",
    "bad_day_at_work": "after a bad day at work",
    "birthday": "{possessive} birthday",
    "normal_evening": "a normal evening",
    "rainy_weekend": "a rainy weekend",
}

#: Closing instruction of every system message (Req 5.5).
THINK_INSTRUCTION = (
    "Think briefly inside <think>...</think>, then call the tools that make {name} happiest, "
    "or answer in plain text if no tool fits."
)


def _or_list(values: Sequence[str]) -> str:
    """('flowers', 'plush_toy') -> 'flowers or plush toy'."""
    return " or ".join(_readable(v) for v in values)


def system_message(scenario: Scenario) -> str:
    """The Partner_State system message, ending with the `<think>` instruction.

    Example:
        You are helping someone make their partner Mia happy.
        Mia (she/her) is feeling sad. Love meter: 40/100. Occasion: her birthday.
        What Mia likes: favorite gifts are flowers or chocolate; she enjoys comedy
        movies and does not enjoy horror movies; her favorite food is pasta; her
        love language is quality time.
        Think briefly inside <think>...</think>, then call the tools ...
    """
    name = scenario.partner_name
    forms = PRONOUN_FORMS[scenario.pronoun]
    state = scenario.state
    prefs = state.preferences
    occasion = OCCASION_PHRASES[state.occasion].format(**forms)
    likes = "; ".join(
        [
            f"favorite gifts are {_or_list(prefs.favorite_gift_types)}",
            f"{forms['pronoun']} enjoys {_or_list(prefs.liked_genres)} movies and does not enjoy "
            f"{_or_list(prefs.disliked_genres)} movies",
            f"{forms['possessive']} favorite food is {_readable(prefs.favorite_food)}",
            f"{forms['possessive']} love language is {_readable(prefs.love_language)}",
        ]
    )
    return "\n".join(
        [
            f"You are helping someone make their partner {name} happy.",
            f"{name} ({forms['pronoun']}/{forms['object']}) is feeling {state.mood}. "
            f"Love meter: {state.love_meter}/100. Occasion: {occasion}.",
            f"What {name} likes: {likes}.",
            THINK_INSTRUCTION.format(name=name),
        ]
    )


def _render(tokenizer: Any, messages: list[dict], tools: Optional[list[dict]]) -> str:
    """Render `messages` to text with the tokenizer's chat template."""
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def _first_diff(a: str, b: str) -> int:
    """Index of the first differing character (len of the shorter if one is a prefix)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def inline_tools_into_system(
    system_text: str, schemas: list[dict], tokenizer: Any, user_text: str = "Hello"
) -> str:
    """Return `system_text` with the template's tool block written into it.

    Renders `[system]` with `tools=schemas` and extracts the system block
    between `<|im_start|>system\\n` and the first `<|im_end|>`. Then checks that
    `[inlined system, user]` without tools renders exactly like
    `[original system, user]` with tools.

    Raises:
        ValueError: If the template has no Qwen system block, or the two
            renderings differ (the message gives the first differing offset).
    """
    rendered = _render(tokenizer, [{"role": "system", "content": system_text}], schemas)
    start = rendered.find(SYSTEM_START)
    end = rendered.find(TURN_END, start + len(SYSTEM_START)) if start >= 0 else -1
    if start < 0 or end < 0:
        raise ValueError(
            "Chat template output has no '<|im_start|>system' block; "
            "use a Qwen2.5/Qwen3 tokenizer for inlined tools."
        )
    inlined = rendered[start + len(SYSTEM_START) : end]

    user = {"role": "user", "content": user_text}
    expected = _render(tokenizer, [{"role": "system", "content": system_text}, user], schemas)
    actual = _render(tokenizer, [{"role": "system", "content": inlined}, user], None)
    if actual != expected:
        offset = _first_diff(actual, expected)
        raise ValueError(
            f"Inlined-tool rendering differs from apply_chat_template(..., tools=schemas) "
            f"at character offset {offset}: inlined {actual[offset:offset + 40]!r} vs "
            f"expected {expected[offset:offset + 40]!r}. Use a Qwen2.5/Qwen3 chat template."
        )
    return inlined


def render_prompt_messages(
    scenario: Scenario,
    tool_names: Optional[Iterable[str]] = None,
    tokenizer: Any = None,
    inline_tools: bool = False,
) -> list[dict]:
    """The conversational prompt `[system, user]` for one Scenario.

    With `inline_tools=True` (GRPO/RLOO) the tool block for `tool_names`
    (default: all nine tools) is written into the system message, which needs
    `tokenizer`. Otherwise the plain Partner_State system message is used and
    the tools travel separately in the `tools` column (SFT/DPO/eval/app).
    """
    system = system_message(scenario)
    if inline_tools:
        if tokenizer is None:
            raise ValueError("render_prompt_messages(inline_tools=True) needs a tokenizer.")
        schemas = tool_schemas(tool_names)
        system = inline_tools_into_system(system, schemas, tokenizer, user_text=scenario.request)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": scenario.request},
    ]


def check_template_has_tools(text: str, tools: Iterable[Union[str, dict]]) -> None:
    """Raise if a rendered prompt does not mention every tool name (Req 5.17).

    `tools` holds tool names or JSON schemas (`{"type": "function", "function": {"name": ...}}`).

    Raises:
        ValueError: `Chat template rendering is missing tool '<name>'.`
    """
    for tool in tools:
        if isinstance(tool, dict):
            name = (tool.get("function") or tool).get("name")
        else:
            name = tool
        if not name or name not in text:
            raise ValueError(f"Chat template rendering is missing tool '{name}'.")


# =============================================================================
# 12. Tool_Dataset_Builder (SFT data)
# =============================================================================
# SFT data uses TRL's tool-calling dataset format
# (https://huggingface.co/docs/trl/main/en/dataset_formats#tool-calling).
# Each row has two columns:
#
#   messages: the conversation. An assistant turn that calls tools carries a
#             `tool_calls` list whose `arguments` are real dicts (not JSON
#             strings); each tool result is a `{"role": "tool"}` message.
#   tools:    the JSON schemas of the tools offered in this conversation
#             (made by `transformers.utils.get_json_schema`, see section 5).
#
# A tiny example:
#
#   {"messages": [
#      {"role": "system", "content": "...partner state..."},
#      {"role": "user", "content": "It's Mia's birthday today."},
#      {"role": "assistant", "content": "<think>Mia loves flowers.</think>",
#       "tool_calls": [{"type": "function",
#                       "function": {"name": "give_gift", "arguments": {"item": "roses"}}}]},
#      {"role": "tool", "name": "give_gift",
#       "content": "{\"result\": \"...\", \"love_delta\": 20, \"love_meter\": 60}"},
#      {"role": "assistant", "content": "I gave Mia the roses. Her love meter is now 60/100."}],
#    "tools": [{"type": "function", "function": {"name": "give_gift", ...}}, ...]}
#
# SFTTrainer passes `tools` to `apply_chat_template`, and the Qwen template
# turns the tool calls into `<tool_call>{...}</tool_call>` text for us.
#
# The think text and free-text arguments come from small, wholesome template
# pools below, so the model sees varied but always kind wording.

#: Default share of public function-calling examples in the final SFT set.
PUBLIC_FRACTION: float = 0.30

#: Why a public example was dropped (Req 5.14).
DROP_REASONS: tuple[str, ...] = ("malformed_json", "arguments_not_object", "unknown_tool", "too_long")

#: A drop reason is one of DROP_REASONS (returned instead of an example).
DropReason = str

#: Readable gift-type phrases for think sentences ("she loves books").
GIFT_TYPE_PHRASES: dict[str, str] = {
    "book": "books",
    "chocolate": "chocolate",
    "flowers": "flowers",
    "gadget": "gadgets",
    "home": "cozy things for the home",
    "jewelry": "jewelry",
    "plush_toy": "plush toys",
}

#: Think sentences per tool and reason. Placeholders: {name}, {pronoun},
#: {Pronoun}, {object}, {possessive}, {Possessive}, {mood}, {occasion},
#: {item}, {gift_type}, {genre}, {dish}, {activity}, {love_language}.
THINK_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "apologize": {
        "after_argument": [
            "We argued earlier, so I should apologize first.",
            "We had a disagreement, so a sincere apology has to come first.",
        ],
        "default": ["A short, sincere apology shows {name} I care."],
    },
    "give_gift": {
        "favorite": [
            "{Name} loves {gift_type}, so the {item} will make {object} smile.",
            "{Possessive} favorite gifts are {gift_type}, so I will give the {item}.",
        ],
        "love_language": ["{Possessive} love language is gifts, so a small {item} fits."],
        "default": ["A small gift like the {item} is a kind gesture for {occasion}."],
    },
    "watch_movie": {
        "liked": [
            "{Pronoun} enjoys {genre} movies, so a {genre} movie night fits.",
            "A {genre} movie is a good pick because {name} likes that genre.",
        ],
        "love_language": ["{Possessive} love language is quality time, so a movie together fits."],
        "default": ["Watching a movie together is a cozy idea for {occasion}."],
    },
    "cook_dinner": {
        "favorite": [
            "{Possessive} favorite food is {dish}, so I will cook it.",
            "{Name} loves {dish}, so cooking it for {object} is a good idea.",
        ],
        "love_language": ["{Possessive} love language is acts of service, so cooking dinner fits."],
        "default": ["Cooking dinner is a caring thing to do for {occasion}."],
    },
    "hug": {
        "comfort": [
            "{Pronoun} is feeling {mood}, so a warm hug will comfort {object}.",
            "Since {name} feels {mood}, a hug is a gentle way to help.",
        ],
        "love_language": ["{Possessive} love language is physical touch, so a hug fits."],
        "default": ["A warm hug is a simple way to show I care."],
    },
    "kiss": {
        "happy": ["{Pronoun} is feeling happy, so a light kiss on the cheek fits the mood."],
        "love_language": ["{Possessive} love language is physical touch, so a light kiss fits."],
        "default": ["A light kiss on the cheek is a sweet gesture."],
    },
    "plan_date": {
        "special": [
            "It is {occasion}, so planning a {activity} date makes the day special.",
            "Since it is {occasion}, a {activity} date would be a lovely surprise.",
        ],
        "love_language": ["{Possessive} love language is quality time, so a {activity} date fits."],
        "default": ["A {activity} date is a nice plan for {occasion}."],
    },
    "send_message": {
        "love_language": [
            "{Possessive} love language is words of affirmation, so a kind message fits.",
            "{Name} values kind words, so I will send a sweet message.",
        ],
        "default": ["A kind message will remind {name} that I care."],
    },
    "ask_how_partner_feels": {
        "bad_day": [
            "{Pronoun} had a bad day at work, so I should ask how {pronoun} feels.",
            "Work was rough for {name}, so listening to {object} comes first.",
        ],
        "default": ["Asking how {name} feels shows I am paying attention."],
    },
}

#: Apology reasons per occasion (free text for `apologize(reason)`).
APOLOGY_REASONS: dict[str, list[str]] = {
    "after_argument": [
        "I'm sorry I raised my voice earlier.",
        "I'm sorry I didn't listen to you during our disagreement.",
        "I'm sorry for what I said earlier. You deserve better.",
        "I'm sorry I got upset. Your feelings matter to me.",
    ],
    "anniversary": ["I'm sorry I've been so busy lately. Today is all about us."],
    "bad_day_at_work": ["I'm sorry you had such a hard day, and that I wasn't more help."],
    "birthday": ["I'm sorry I wasn't around much this week. Today is all about you."],
    "normal_evening": ["I'm sorry I've been distracted lately."],
    "rainy_weekend": ["I'm sorry I've been grumpy about the weather."],
}

#: Kind message texts per occasion (free text for `send_message(text)`).
MESSAGE_TEXTS: dict[str, list[str]] = {
    "after_argument": [
        "I care about you a lot, and I want us to be okay.",
        "Thank you for being patient with me. I love you.",
    ],
    "anniversary": [
        "Happy anniversary! Every year with you is my favorite.",
        "Thank you for another wonderful year together.",
    ],
    "bad_day_at_work": [
        "I'm proud of you for getting through today. I'm here for you.",
        "You worked so hard today. Time to rest, I've got you.",
    ],
    "birthday": [
        "Happy birthday! I'm so lucky to have you.",
        "Happy birthday to my favorite person!",
    ],
    "normal_evening": [
        "Just thinking of you and smiling.",
        "You make ordinary evenings feel special.",
    ],
    "rainy_weekend": [
        "Rainy days are my favorite when I spend them with you.",
        "Let's stay cozy together this weekend.",
    ],
}

#: Past-tense action phrases for the final assistant message.
ACTION_PAST: dict[str, str] = {
    "give_gift": "gave {name} the {item}",
    "watch_movie": "watched a {genre} movie together",
    "cook_dinner": "cooked {dish}",
    "hug": "gave {object} a warm hug",
    "kiss": "gave {object} a light kiss on the cheek",
    "apologize": "apologized",
    "plan_date": "planned a {activity} date",
    "send_message": "sent {object} a kind message",
    "ask_how_partner_feels": "asked how {pronoun} feels",
}

#: Final plain-text assistant message after the tool results.
FINAL_TEMPLATES: list[str] = [
    "I {actions}. {Possessive} love meter is now {meter}/100.",
    "Done: I {actions}. {Name}'s love meter went from {start} to {meter}.",
]

#: No_Tool think sentences.
NO_TOOL_THINK: list[str] = [
    "This is a question about {name}'s {topic}, so no tool is needed.",
    "The user only wants to know {name}'s {topic}, so I can answer directly.",
]

#: No_Tool topic phrases per Preferences field.
NO_TOOL_TOPICS: dict[str, str] = {
    "favorite_food": "favorite food",
    "love_language": "love language",
    "liked_genres": "favorite kinds of movies",
    "disliked_genres": "least favorite movie genre",
    "favorite_gift_types": "favorite kinds of gifts",
}

#: No_Tool answers per Preferences field; {values} holds every keyword.
NO_TOOL_ANSWERS: dict[str, str] = {
    "favorite_food": "{Possessive} favorite food is {values}.",
    "love_language": "{Possessive} love language is {values}.",
    "liked_genres": "{Pronoun} enjoys {values} movies.",
    "disliked_genres": "{Pronoun} does not enjoy {values} movies, so it's best to pick another genre.",
    "favorite_gift_types": "{Pronoun} likes {values} gifts best.",
}

#: System message for public examples (they have no partner state).
PUBLIC_SYSTEM_MESSAGE: str = (
    "You are a helpful assistant. Think briefly inside <think>...</think>, then call the "
    "tools that fit the request, or answer in plain text if no tool fits."
)

#: Plain-text answer for public rows with no tool call (irrelevance).
PUBLIC_NO_TOOL_ANSWER: str = (
    "<think>None of the available tools fit this request.</think>"
    "I can't do that with the available tools."
)


def _join_and(items: Sequence[str]) -> str:
    """['a'] -> 'a'; ['a', 'b'] -> 'a and b'; ['a', 'b', 'c'] -> 'a, b, and c'."""
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _text_fields(scenario: Scenario, call: Optional[ToolCall] = None) -> dict[str, str]:
    """Placeholder values for the template pools above."""
    forms = PRONOUN_FORMS[scenario.pronoun]
    state = scenario.state
    fields = {
        "name": scenario.partner_name,
        "Name": scenario.partner_name,
        "Possessive": _capitalize(forms["possessive"]),
        "mood": state.mood,
        "occasion": OCCASION_PHRASES[state.occasion].format(**forms),
        "love_language": _readable(state.preferences.love_language),
        **forms,
    }
    if call is not None:
        args = call.arguments
        item = args.get("item", "")
        fields.update(
            item=_readable(item),
            gift_type=GIFT_TYPE_PHRASES.get(GIFT_ITEMS.get(item, ""), "gifts"),
            genre=_readable(args.get("genre", "")),
            dish=_readable(args.get("dish", "")),
            activity=_readable(args.get("activity", "")),
        )
    return fields


def _think_reason(call: ToolCall, state: PartnerState) -> str:
    """Which THINK_TEMPLATES entry explains `call` best for this partner state."""
    prefs = state.preferences
    args = call.arguments
    strong = {
        "apologize": "after_argument" if state.occasion == "after_argument" else None,
        "give_gift": "favorite" if GIFT_ITEMS.get(args.get("item")) in prefs.favorite_gift_types else None,
        "watch_movie": "liked" if args.get("genre") in prefs.liked_genres else None,
        "cook_dinner": "favorite" if args.get("dish") == prefs.favorite_food else None,
        "hug": "comfort" if state.mood in COMFORT_MOODS else None,
        "kiss": "happy" if state.mood == "happy" else None,
        "plan_date": "special" if state.occasion in ("birthday", "anniversary") else None,
        "ask_how_partner_feels": "bad_day" if state.occasion == "bad_day_at_work" else None,
    }.get(call.name)
    if strong:
        return strong
    if call.name in LOVE_LANGUAGE_TOOLS.get(prefs.love_language, frozenset()):
        return "love_language"
    return "default"


def think_text(scenario: Scenario, calls: Sequence[ToolCall], rng: random.Random) -> str:
    """1 to 3 sentences, one per planned call (at most 3), each citing its motivation."""
    sentences = []
    for call in list(calls)[:MAX_CALLS_PER_TURN]:
        pool = THINK_TEMPLATES[call.name]
        templates = pool.get(_think_reason(call, scenario.state)) or pool["default"]
        sentence = _capitalize(rng.choice(templates).format(**_text_fields(scenario, call)))
        if sentence not in sentences:  # e.g. two "default" hug-like lines
            sentences.append(sentence)
    return " ".join(sentences)


def _with_free_text(call: ToolCall, occasion: str, rng: random.Random) -> ToolCall:
    """Replace the placeholder free text of a planned call with a pool text."""
    if call.name == "apologize":
        return ToolCall(call.name, {"reason": rng.choice(APOLOGY_REASONS[occasion])})
    if call.name == "send_message":
        return ToolCall(call.name, {"text": rng.choice(MESSAGE_TEXTS[occasion])})
    return call


def _tool_call_entry(call: ToolCall) -> dict:
    """TRL `tool_calls` entry; `arguments` stays a dict (Req 5.2)."""
    return {"type": "function", "function": {"name": call.name, "arguments": dict(call.arguments)}}


def _no_tool_topic(scenario: Scenario) -> Optional[str]:
    """The Preferences field whose values are the scenario's keywords."""
    keywords = tuple(scenario.no_tool_keywords or ())
    for field in NO_TOOL_TOPICS:
        values = getattr(scenario.state.preferences, field)
        values = (values,) if isinstance(values, str) else values
        if tuple(normalize_keyword(v) for v in values) == keywords:
            return field
    return None


def _no_tool_answer(scenario: Scenario, rng: random.Random) -> str:
    """`<think>` + plain answer that contains every No_Tool keyword (Req 5.8)."""
    fields = _text_fields(scenario)
    topic = _no_tool_topic(scenario)
    values = _join_and(list(scenario.no_tool_keywords or ()))
    if topic is None:
        think = f"This is a question about {scenario.partner_name}, so no tool is needed."
        answer = f"The answer is {values}."
    else:
        think = rng.choice(NO_TOOL_THINK).format(topic=NO_TOOL_TOPICS[topic], **fields)
        answer = NO_TOOL_ANSWERS[topic].format(values=values, **fields)
    return f"{THINK_OPEN}{think}{THINK_CLOSE}{answer}"


def build_sim_sft_example(
    scenario: Scenario, sim: Optional[RelationshipSim] = None, rng: Optional[random.Random] = None
) -> dict:
    """One TRL tool-format SFT row (`messages`, `tools`) for a Relationship_Sim Scenario.

    Tool scenarios: system, user, assistant (`<think>` + Optimal_Plan
    `tool_calls`), one `tool` message per call with the simulator's JSON result,
    and a final plain-text assistant message. No_Tool_Scenarios: system, user,
    and one assistant message with `<think>` + an answer containing every keyword.
    """
    sim = sim or RelationshipSim()
    rng = rng or random.Random(0)
    messages: list[dict] = render_prompt_messages(scenario)
    tools = tool_schemas()

    if scenario.is_no_tool:
        messages.append({"role": "assistant", "content": _no_tool_answer(scenario, rng)})
        return {"messages": messages, "tools": tools}

    occasion = scenario.state.occasion
    calls = [_with_free_text(c, occasion, rng) for group in scenario.optimal_plan for c in group]
    messages.append(
        {
            "role": "assistant",
            "content": f"{THINK_OPEN}{think_text(scenario, calls, rng)}{THINK_CLOSE}",
            "tool_calls": [_tool_call_entry(c) for c in calls],
        }
    )
    episode = sim.new_episode(scenario)
    for call in calls:
        result = episode.execute(call)
        messages.append(
            {"role": "tool", "name": call.name, "content": json.dumps(result, ensure_ascii=False)}
        )

    actions = [ACTION_PAST[c.name].format(**_text_fields(scenario, c)) for c in calls]
    final = rng.choice(FINAL_TEMPLATES).format(
        actions=_join_and(actions),
        meter=episode.love_meter,
        start=episode.initial_meter,
        **_text_fields(scenario),
    )
    messages.append({"role": "assistant", "content": final})
    return {"messages": messages, "tools": tools}


# -----------------------------------------------------------------------------
# Public function-calling data (argilla/apigen-function-calling, xLAM style)
# -----------------------------------------------------------------------------
# Rows look like:
#   query:   "What is the weather in Paris?"
#   tools:   '[{"name": "get_weather", "description": "...",
#              "parameters": {"city": {"type": "str", "description": "..."}}}]'
#   answers: '[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
# We convert the tools to OpenAI-style JSON schemas and the answers to one
# assistant turn with structured `tool_calls` (no tool results exist for them).

#: xLAM type names -> JSON schema types.
_XLAM_TYPES: dict[str, str] = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "tuple": "array",
    "set": "array",
    "dict": "object",
    "object": "object",
}


def _json_type(xlam_type: Any) -> str:
    """'List[int], optional' -> 'array'; unknown types fall back to 'string'."""
    text = str(xlam_type or "").split(",")[0].strip().lower()
    base = text.split("[")[0].strip()
    return _XLAM_TYPES.get(base, "string")


def _convert_public_tool(tool: Any) -> Optional[dict]:
    """One xLAM (or already OpenAI-style) tool spec -> OpenAI schema; None if malformed."""
    if not isinstance(tool, dict):
        return None
    if isinstance(tool.get("function"), dict):  # already {"type": "function", "function": {...}}
        tool = tool["function"]
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        return None
    params = tool.get("parameters") or {}
    if not isinstance(params, dict):
        return None
    if params.get("type") == "object" and isinstance(params.get("properties"), dict):
        parameters = copy.deepcopy(params)  # already a JSON schema
    else:
        properties, required = {}, []
        for arg, spec in params.items():
            spec = spec if isinstance(spec, dict) else {}
            properties[arg] = {
                "type": _json_type(spec.get("type")),
                "description": str(spec.get("description", "")),
            }
            if "default" not in spec and "optional" not in str(spec.get("type", "")).lower():
                required.append(arg)
        parameters = {"type": "object", "properties": properties, "required": required}
    return {
        "type": "function",
        "function": {"name": name, "description": str(tool.get("description", "")), "parameters": parameters},
    }


def _load_json_list(value: Any) -> Optional[list]:
    """A JSON-string (or already decoded) list, or None if it is not one."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return None
    return value if isinstance(value, list) else None


def _char_length(example: dict) -> int:
    """Default length measure: characters of the example's JSON (no tokenizer needed)."""
    return len(json.dumps(example, ensure_ascii=False))


def token_length_fn(tokenizer: Any) -> Callable[[dict], int]:
    """A length measure in tokens: renders `messages` + `tools` with the chat template."""

    def length(example: dict) -> int:
        text = tokenizer.apply_chat_template(
            example["messages"], tools=decode_tools(example.get("tools")), tokenize=False
        )
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return length


def convert_public_example(
    row: dict,
    max_length: Optional[int] = None,
    length_fn: Optional[Callable[[dict], int]] = None,
) -> Union[dict, DropReason]:
    """Convert one xLAM-style row (`query`, `tools`, `answers`) to a TRL tool-format row.

    Returns the example dict, or a drop reason string from DROP_REASONS:
    `malformed_json` (tools/answers are not valid JSON lists of objects),
    `arguments_not_object`, `unknown_tool` (answer names a tool missing from
    `tools`), or `too_long` (`length_fn(example) > max_length`; the default
    `length_fn` counts characters, pass `token_length_fn(tokenizer)` for tokens).
    """
    query = row.get("query") if isinstance(row, dict) else None
    raw_tools = _load_json_list(row.get("tools")) if isinstance(row, dict) else None
    answers = _load_json_list(row.get("answers")) if isinstance(row, dict) else None
    if not isinstance(query, str) or raw_tools is None or answers is None:
        return "malformed_json"
    tools = [_convert_public_tool(t) for t in raw_tools]
    if any(t is None for t in tools):
        return "malformed_json"
    known = {t["function"]["name"] for t in tools}

    calls = []
    for answer in answers:
        if not isinstance(answer, dict) or not isinstance(answer.get("name"), str):
            return "malformed_json"
        arguments = answer.get("arguments", {})
        if not isinstance(arguments, dict):
            return "arguments_not_object"
        if answer["name"] not in known:
            return "unknown_tool"
        calls.append(ToolCall(answer["name"], arguments))

    messages = [
        {"role": "system", "content": PUBLIC_SYSTEM_MESSAGE},
        {"role": "user", "content": query},
    ]
    if calls:
        names = list(dict.fromkeys(c.name for c in calls))  # unique, in order
        messages.append(
            {
                "role": "assistant",
                "content": f"{THINK_OPEN}The user needs {_join_and(names)}.{THINK_CLOSE}",
                "tool_calls": [_tool_call_entry(c) for c in calls],
            }
        )
    else:
        messages.append({"role": "assistant", "content": PUBLIC_NO_TOOL_ANSWER})
    example = {"messages": messages, "tools": tools}

    if max_length is not None and (length_fn or _char_length)(example) > max_length:
        return "too_long"
    return example


def convert_public_examples(
    rows: Iterable[dict],
    max_length: Optional[int] = None,
    length_fn: Optional[Callable[[dict], int]] = None,
) -> tuple[list[dict], Counter]:
    """Convert many public rows; returns (examples, Counter of drop reasons)."""
    examples, dropped = [], Counter()
    for row in rows:
        result = convert_public_example(row, max_length=max_length, length_fn=length_fn)
        if isinstance(result, str):
            dropped[result] += 1
        else:
            examples.append(result)
    return examples, dropped


def public_mix_count(n_sim: int, n_public: int, fraction: float) -> int:
    """How many public rows to mix in so they make up `fraction` of the final set.

    `min(n_public, round(fraction * n_sim / (1 - fraction)))`; 0 when the
    fraction is not positive, all public rows when it is 1 or more.
    """
    if fraction <= 0 or n_public <= 0:
        return 0
    if fraction >= 1:
        return n_public
    return min(n_public, round(fraction * n_sim / (1 - fraction)))


def build_sft_examples(
    scenarios: Sequence[Scenario],
    sim: Optional[RelationshipSim] = None,
    public_examples: Optional[Sequence[dict]] = None,
    public_fraction: float = PUBLIC_FRACTION,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    """SFT rows for every Scenario, plus a seeded sample of converted public rows.

    `public_examples` are rows already converted by `convert_public_example`
    (pass `None` or `[]` for Relationship_Sim data only). The result is shuffled
    with `rng`.
    """
    sim = sim or RelationshipSim()
    rng = rng or random.Random(sim.seed)
    rows = [build_sim_sft_example(s, sim, rng) for s in scenarios]
    public = list(public_examples or [])
    k = public_mix_count(len(rows), len(public), public_fraction)
    rows += [copy.deepcopy(r) for r in rng.sample(public, k)]
    rng.shuffle(rows)
    return rows


# -----------------------------------------------------------------------------
# Dataset construction and public data loading (lazy `datasets` import)
# -----------------------------------------------------------------------------


def _version_tuple(version: str) -> tuple[int, ...]:
    """'4.8.5' -> (4, 8, 5); '4.7.0.dev0' -> (4, 7, 0)."""
    parts = []
    for piece in version.split(".")[:3]:
        digits = "".join(itertools.takewhile(str.isdigit, piece))
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (3 - len(parts)))


def decode_tools(value: Any) -> Any:
    """`tools` column value as a list of schemas (decodes the `json.dumps` form)."""
    return json.loads(value) if isinstance(value, str) else value


def make_dataset(rows: Sequence[dict]) -> Any:
    """Build a `datasets.Dataset` from tool-format rows.

    With `datasets>=4.7.0` the mixed-shape columns (`tools`, tool-call
    arguments) are stored as JSON via `on_mixed_types="use_json"` (Req 5.10).
    Older versions get the `tools` column as a `json.dumps` string; TRL's SFT
    and DPO trainers `json.loads` it back, and `decode_tools` does the same
    for our own rendering (Req 5.11).
    """
    import datasets  # lazy: the module stays stdlib-only at import time

    rows = list(rows)
    if _version_tuple(datasets.__version__) >= (4, 7, 0):
        return datasets.Dataset.from_list(rows, on_mixed_types="use_json")
    serialized = [
        {**row, "tools": json.dumps(row["tools"])}
        if "tools" in row and not isinstance(row["tools"], str)
        else row
        for row in rows
    ]
    return datasets.Dataset.from_list(serialized)


def load_public_dataset(dataset_id: Optional[str], split: str = "train") -> Any:
    """Load the public function-calling dataset, or return None (Req 5.13).

    `None` as the id disables public data. Any load error prints a warning
    with the id and the error, and training continues with simulator data only.
    """
    if not dataset_id:
        return None
    try:
        import datasets  # lazy import

        return datasets.load_dataset(dataset_id, split=split)
    except Exception as exc:  # noqa: BLE001 - any failure falls back to sim-only data
        print(
            f"WARNING: could not load public dataset '{dataset_id}': {type(exc).__name__}: {exc}. "
            "Continuing with Relationship_Sim data only."
        )
        return None


# =============================================================================
# 13. RL prompts (GRPO / RLOO)
# =============================================================================
# GRPO and RLOO generate their own completions, so a row only needs the
# question, not an answer:
#
#   prompt:   [system, user]. The system message already contains the tool
#             block. GRPOTrainer and RLOOTrainer (TRL v1.12.0) do not read the
#             `tools` column when they render prompts, so without this the
#             model would be asked to call tools it has never been shown.
#             Inlining gives exactly the text SFT and DPO see via `tools=`.
#   tools:    the JSON schemas as a JSON string. Not used for rendering here;
#             it reaches `tool_reward` as a keyword argument (which tools were
#             offered).
#   scenario: `Scenario.to_json()`, so `tool_reward` can rebuild the Scenario
#             and score each completion with the simulator.
#
# The same rows feed both GRPO and RLOO, so the two trainers see identical
# prompts (Req 9.4).

#: Inlined tool block (the text the template appends after the system text),
#: cached per (tokenizer, tool-name set). Inlining calls apply_chat_template
#: three times, so we do the full, checked inlining once per tool set.
_TOOL_BLOCK_CACHE: dict[tuple[int, tuple[str, ...]], Optional[str]] = {}


def _rl_system_text(scenario: Scenario, names: tuple[str, ...], tokenizer: Any) -> str:
    """System message with the tool block inlined (cached tool block when possible)."""
    key = (id(tokenizer), names)
    system = system_message(scenario)
    if key not in _TOOL_BLOCK_CACHE:
        inlined = render_prompt_messages(scenario, names, tokenizer, inline_tools=True)[0]["content"]
        # Qwen templates write `system text + tool block`; cache the block only
        # if that holds, otherwise fall back to full inlining for every row.
        _TOOL_BLOCK_CACHE[key] = inlined[len(system):] if inlined.startswith(system) else None
        return inlined
    block = _TOOL_BLOCK_CACHE[key]
    if block is None:
        return render_prompt_messages(scenario, names, tokenizer, inline_tools=True)[0]["content"]
    return system + block


def build_rl_examples(
    scenarios: Sequence[Scenario],
    tokenizer: Any,
    tool_names: Optional[Iterable[str]] = None,
) -> list[dict]:
    """GRPO/RLOO rows: `prompt` (tools inlined), `tools` (JSON string), `scenario` (JSON).

    `tool_names` defaults to all nine Partner_Tools in `PARTNER_TOOLS` order
    (the same order SFT and DPO use, so the tool block text matches). The first row is checked
    to mention every tool name (Req 5.17).

    Raises:
        ValueError: If the tokenizer's template cannot inline tools, or the
            rendered prompt is missing a tool name.
    """
    names = tuple(TOOLS_BY_NAME if tool_names is None else tool_names)
    schemas = tool_schemas(names)
    tools_json = json.dumps(schemas, ensure_ascii=False)
    rows: list[dict] = []
    for scenario in scenarios:
        prompt = [
            {"role": "system", "content": _rl_system_text(scenario, names, tokenizer)},
            {"role": "user", "content": scenario.request},
        ]
        if not rows:
            check_template_has_tools(prompt[0]["content"], schemas)
        rows.append({"prompt": prompt, "tools": tools_json, "scenario": scenario.to_json()})
    return rows


# =============================================================================
# 14. Preference_Builder (DPO pairs)
# =============================================================================
# DPO learns from pairs of answers to the same prompt: a better one ("chosen")
# and a worse one ("rejected"). It nudges the model to make the chosen answer
# more likely and the rejected one less likely. No reward is computed during
# DPO training, so we pick the pairs up front with the same Tool_Reward that
# GRPO uses. That way both methods aim at the same goal.
#
# For every prompt we write a few candidate answers: the best plan from the
# simulator, plus copies with one deliberate mistake (a wrong argument, a
# disliked movie, a forgotten apology, an extra call, no <think>, or a broken
# <tool_call> block). We score all candidates, and pair the best one with
# every candidate that scores strictly lower. Candidates that score the same
# as the best teach nothing, so they are dropped and counted as `tie_dropped`.
#
# Row format (TRL conversational preference format):
#   prompt:   [system, user], plain system text. DPOTrainer passes `tools` to
#             the chat template itself, giving the same prompt text as GRPO.
#   chosen:   [{"role": "assistant", "content": "<think>...</think>\n<tool_call>..."}]
#   rejected: [{"role": "assistant", "content": ...}]
#   tools:    the JSON schemas of all nine Partner_Tools.

#: Mistakes for tool scenarios (Req 6.2).
TOOL_PERTURBATIONS: tuple[str, ...] = (
    "wrong_argument",
    "disliked_genre",
    "missing_apologize",
    "extra_call",
    "missing_think",
    "malformed_block",
)

#: Mistakes for No_Tool_Scenarios.
NO_TOOL_PERTURBATIONS: tuple[str, ...] = ("missing_keyword", "spurious_call", "missing_think")

#: Candidates per prompt; the notebook sets this to GRPO_NUM_GENERATIONS (Req 9.5).
DEFAULT_N_CANDIDATES: int = 4


def _tool_text(think: Optional[str], calls: Sequence[ToolCall]) -> str:
    """Raw assistant text: optional `<think>` block, then one Tool_Call_Block per call."""
    parts = [] if think is None else [f"{THINK_OPEN}{think}{THINK_CLOSE}"]
    if calls:
        parts.append(print_tool_calls(calls))
    return "\n".join(parts)


def _break_block(text: str, index: int) -> Optional[str]:
    """Remove the closing `}` of block `index`'s JSON, so that block is invalid JSON.

    Returns None if `text` has no complete block number `index`.
    """
    pos = -1
    for _ in range(index + 1):
        pos = text.find(TOOL_CALL_OPEN, pos + 1)
        if pos == -1:
            return None
    end = text.find(TOOL_CALL_CLOSE, pos)
    brace = text.rfind("}", pos, end) if end != -1 else -1
    if brace == -1:
        return None
    return text[:brace] + text[brace + 1:]


def _strip_think(text: str) -> str:
    """Drop the leading `<think>...</think>` block (and the newline after it)."""
    start = text.find(THINK_OPEN)
    end = text.find(THINK_CLOSE, start)
    if start == -1 or end == -1:
        return text
    return (text[:start] + text[end + len(THINK_CLOSE):]).lstrip("\n")


def _perturbed_calls(
    kind: str, scenario: Scenario, calls: list[ToolCall], rng: random.Random
) -> Optional[list[ToolCall]]:
    """Calls with one mistake of type `kind`, or None if it does not apply here."""
    state = scenario.state
    occasion = state.occasion
    if kind == "wrong_argument":
        options = [
            i for i, c in enumerate(calls)
            for arg in TOOL_ARGS.get(c.name, ()) if (c.name, arg) in CATALOG_ARGS
        ]
        if not options:
            return None
        i = rng.choice(options)
        call = calls[i]
        arg = next(a for a in TOOL_ARGS[call.name] if (call.name, a) in CATALOG_ARGS)
        values = [v for v in CATALOG_ARGS[(call.name, arg)] if v != call.arguments.get(arg)]
        if not values:
            return None
        return calls[:i] + [ToolCall(call.name, {arg: rng.choice(values)})] + calls[i + 1:]
    if kind == "disliked_genre":
        if not state.preferences.disliked_genres:
            return None
        bad = ToolCall("watch_movie", {"genre": rng.choice(list(state.preferences.disliked_genres))})
        movies = [i for i, c in enumerate(calls) if c.name == "watch_movie"]
        if movies:
            i = movies[0]
            return calls[:i] + [bad] + calls[i + 1:]
        return calls + [bad]
    if kind == "missing_apologize":
        if occasion == "after_argument" and any(c.name == "apologize" for c in calls):
            return [c for c in calls if c.name != "apologize"]
        return calls[1:] if calls else None
    if kind == "extra_call":
        planned = {c.key() for c in calls}
        extras = [a for a in action_space() if a.key() not in planned]
        if not extras:
            return None
        return calls + [_with_free_text(rng.choice(extras), occasion, rng)]
    raise ValueError(f"Unknown perturbation {kind!r}")


def _tool_candidates(
    scenario: Scenario, n_candidates: int, rng: random.Random
) -> list[tuple[str, str]]:
    """(kind, text) candidates for a tool scenario: optimal first, then perturbations."""
    occasion = scenario.state.occasion
    calls = [_with_free_text(c, occasion, rng) for group in scenario.optimal_plan for c in group]
    optimal = _tool_text(think_text(scenario, calls, rng), calls)
    candidates = [("optimal", optimal)]
    kinds = list(TOOL_PERTURBATIONS)
    rng.shuffle(kinds)  # seeded choice of which mistakes this prompt gets
    for kind in kinds:
        if len(candidates) >= n_candidates:
            break
        if kind == "missing_think":
            text = _strip_think(optimal)
        elif kind == "malformed_block":
            text = _break_block(optimal, rng.randrange(len(calls))) if calls else None
        else:
            bad = _perturbed_calls(kind, scenario, calls, rng)
            # Think text matches the (wrong) calls, as a model's would.
            text = None if bad is None else _tool_text(think_text(scenario, bad, rng) or "No action is needed.", bad)
        if text is not None and all(text != t for _, t in candidates):
            candidates.append((kind, text))
    return candidates


def _no_tool_candidates(
    scenario: Scenario, n_candidates: int, rng: random.Random
) -> list[tuple[str, str]]:
    """(kind, text) candidates for a No_Tool_Scenario: optimal first, then mistakes."""
    optimal = _no_tool_answer(scenario, rng)
    think_end = optimal.find(THINK_CLOSE) + len(THINK_CLOSE)
    head, answer = optimal[:think_end], optimal[think_end:]
    keywords = list(scenario.no_tool_keywords or ())
    candidates = [("optimal", optimal)]
    kinds = list(NO_TOOL_PERTURBATIONS)
    rng.shuffle(kinds)
    for kind in kinds:
        if len(candidates) >= n_candidates:
            break
        if kind == "missing_keyword":
            hits = [k for k in keywords if k in answer]
            text = head + answer.replace(rng.choice(hits), "something") if hits else None
        elif kind == "spurious_call":
            text = optimal + "\n" + print_tool_calls([ToolCall("ask_how_partner_feels", {})])
        else:  # missing_think
            text = answer.strip()
        if text is not None and all(text != t for _, t in candidates):
            candidates.append((kind, text))
    return candidates


def build_preference_pairs(
    scenarios: Sequence[Scenario],
    sim: Optional[RelationshipSim] = None,
    n_candidates: int = DEFAULT_N_CANDIDATES,
    rng: Optional[random.Random] = None,
) -> tuple[list[dict], dict]:
    """DPO rows (`prompt`, `chosen`, `rejected`, `tools`) and a summary of counts.

    Each Scenario gets up to `n_candidates` candidates (the optimal answer plus
    `n_candidates - 1` distinct, seeded perturbations). Candidates are ranked by
    `score_completion(...).total`; the best one is paired with every candidate
    scoring strictly lower, so `score(chosen) > score(rejected)` always holds
    (Req 6.3, 6.5). Candidates tied with the best are dropped and counted (Req 6.4).

    Summary keys: `prompts`, `candidates`, `pairs`, `tie_dropped`, and
    `per_perturbation` = {kind: {"candidates", "pairs", "tie_dropped"}}.

    Raises:
        ValueError: If `n_candidates < 2` (a pair needs two answers).
    """
    if n_candidates < 2:
        raise ValueError(f"n_candidates must be >= 2, got {n_candidates}")
    sim = sim or RelationshipSim()  # kept for API symmetry; scoring uses the shared scorer
    rng = rng or random.Random(0)
    tools = tool_schemas()
    names = [_schema_name(s) for s in tools]

    kinds = ("optimal",) + TOOL_PERTURBATIONS + tuple(k for k in NO_TOOL_PERTURBATIONS if k not in TOOL_PERTURBATIONS)
    per_kind = {k: {"candidates": 0, "pairs": 0, "tie_dropped": 0} for k in kinds}
    pairs: list[dict] = []
    n_candidates_total = 0
    tie_dropped = 0

    for scenario in scenarios:
        make = _no_tool_candidates if scenario.is_no_tool else _tool_candidates
        candidates = make(scenario, n_candidates, rng)
        scored = [(score_completion(text, scenario, names).total, kind, text) for kind, text in candidates]
        n_candidates_total += len(scored)
        # max() keeps the first of equal scores, so the optimal answer wins ties.
        best_index = max(range(len(scored)), key=lambda i: scored[i][0])
        best_score, _, best_text = scored[best_index]
        prompt = render_prompt_messages(scenario)
        for i, (score, kind, text) in enumerate(scored):
            per_kind[kind]["candidates"] += 1
            if i == best_index:
                continue
            if score < best_score:
                per_kind[kind]["pairs"] += 1
                pairs.append(
                    {
                        "prompt": [dict(m) for m in prompt],
                        "chosen": [{"role": "assistant", "content": best_text}],
                        "rejected": [{"role": "assistant", "content": text}],
                        "tools": copy.deepcopy(tools),
                    }
                )
            else:
                per_kind[kind]["tie_dropped"] += 1
                tie_dropped += 1

    summary = {
        "prompts": len(scenarios),
        "candidates": n_candidates_total,
        "pairs": len(pairs),
        "tie_dropped": tie_dropped,
        "per_perturbation": per_kind,
    }
    return pairs, summary


# =============================================================================
# 15. BFCL eval builder and data summary
# =============================================================================
# The benchmark asks every model the same held-out questions. Each BFCL
# category mixes two sources (about half each):
#
#   sim     built from eval-split Scenarios (never seen in training):
#             simple       one tool offered, one correct call
#             multiple     2 to 4 tools offered (the best one among them), one call
#             parallel     all nine tools, Scenarios whose best plan has 2+ calls
#             irrelevance  No_Tool questions, plus out-of-scope requests
#                          ("book a flight") that no partner tool can do
#   public  rows of the public function-calling dataset that were NOT used for
#           SFT, sorted by shape: 1 tool + 1 answer -> simple, 2 to 4 tools +
#           1 answer -> multiple, 2+ answers -> parallel, no answer -> irrelevance.
#
# The extra `sim` key holds every eval Scenario with all nine tools; it feeds
# `Sim Task Success` and `Mean Love Gain`.
#
# If the public dataset could not be loaded, pass `public_eval=None`: the
# categories then hold Relationship_Sim items only (Req 10.14).

#: Eval examples per BFCL category (sim + public).
N_PER_CATEGORY: int = 50

#: Fewest eval examples a BFCL category may have (Req 5.16).
MIN_PER_CATEGORY: int = 20

#: Requests no Partner_Tool can do; the correct answer calls no tool.
OUT_OF_SCOPE_REQUESTS: list[str] = [
    "Can you book a flight to Tokyo for me next Friday?",
    "What's the weather going to be like in Paris tomorrow?",
    "Please convert 100 US dollars to euros.",
    "Can you order me a taxi to the airport?",
    "Set an alarm for 7 am tomorrow, please.",
    "What is 17 times 23?",
    "Translate 'good morning' into Spanish for me.",
    "Can you check my bank account balance?",
    "Remind me to renew my passport next month.",
    "What time does the train to Osaka leave?",
    "Please reserve a hotel room in London for two nights.",
    "How many kilometers is a marathon?",
]


@dataclass
class EvalItem:
    """One benchmark question with its ground truth.

    `category` is a BFCL category or `sim` (full-toolset Scenario item);
    `gt_calls` is empty for `irrelevance`; `scenario` is set for sim items
    built from a Scenario (None for public and out-of-scope items).
    """

    category: Literal["simple", "multiple", "parallel", "irrelevance", "sim"]
    source: Literal["sim", "public"]
    prompt_messages: list[dict]
    tools: list[dict]
    gt_calls: list[GTCall]
    scenario: Optional[Scenario] = None
    item_id: str = ""


def gt_from_calls(calls: Iterable[ToolCall]) -> list[GTCall]:
    """Ground truth for a list of calls: each value is the only accepted one,
    except Free_Text_Arguments, which accept any non-empty text."""
    return [
        GTCall(
            c.name,
            {k: FreeText() if (c.name, k) in FREE_TEXT_ARGS else [v] for k, v in c.arguments.items()},
        )
        for c in calls
    ]


def _flat_plan(plan: Sequence[Sequence[ToolCall]]) -> list[ToolCall]:
    return [c for group in plan for c in group]


def _sim_item(category: str, scenario: Scenario, names: Sequence[str], calls: Sequence[ToolCall], tag: str) -> EvalItem:
    return EvalItem(
        category=category,
        source="sim",
        prompt_messages=render_prompt_messages(scenario),
        tools=tool_schemas(names),
        gt_calls=gt_from_calls(calls),
        scenario=scenario,
        item_id=f"sim-{category}-{tag}-{scenario.scenario_id[:10]}",
    )


def _sim_bfcl_items(
    sim: RelationshipSim, scenarios: Sequence[Scenario], n: int, rng: random.Random
) -> dict[str, list[EvalItem]]:
    """Up to `n` Relationship_Sim items per BFCL category."""
    all_names = list(TOOLS_BY_NAME)
    tool_scenarios = [s for s in scenarios if not s.is_no_tool]
    items: dict[str, list[EvalItem]] = {c: [] for c in BFCL_CATEGORIES}

    # simple: one tool from the Scenario's best plan, its best single call.
    for s in tool_scenarios[:n]:
        tool = rng.choice(sorted({c.name for c in _flat_plan(s.optimal_plan)}))
        calls = _flat_plan(sim.optimal_plan(s, allowed_tools=[tool], max_calls=1))
        items["simple"].append(_sim_item("simple", s, [tool], calls, tool))

    # multiple: the best single tool plus 1 to 3 distractors, in catalog order.
    for s in tool_scenarios[:n]:
        best = _flat_plan(sim.optimal_plan(s, max_calls=1))[0].name
        others = [t for t in all_names if t != best]
        offered = {best, *rng.sample(others, rng.randint(1, 3))}
        names = [t for t in all_names if t in offered]
        calls = _flat_plan(sim.optimal_plan(s, allowed_tools=names, max_calls=1))
        items["multiple"].append(_sim_item("multiple", s, names, calls, str(len(names))))

    # parallel: all tools, Scenarios whose best plan has 2+ calls.
    multi = [s for s in tool_scenarios if len(_flat_plan(s.optimal_plan)) >= 2]
    for s in multi[:n]:
        items["parallel"].append(_sim_item("parallel", s, all_names, _flat_plan(s.optimal_plan), "all"))

    # irrelevance: No_Tool questions first, then out-of-scope requests.
    for s in [s for s in scenarios if s.is_no_tool][:n]:
        items["irrelevance"].append(_sim_item("irrelevance", s, all_names, [], "no_tool"))
    for i, s in enumerate(tool_scenarios[: max(0, n - len(items["irrelevance"]))]):
        request = OUT_OF_SCOPE_REQUESTS[i % len(OUT_OF_SCOPE_REQUESTS)]
        items["irrelevance"].append(
            EvalItem(
                category="irrelevance",
                source="sim",
                prompt_messages=[
                    {"role": "system", "content": system_message(s)},
                    {"role": "user", "content": request},
                ],
                tools=tool_schemas(all_names),
                gt_calls=[],
                scenario=None,  # the request is not the Scenario's own request
                item_id=f"sim-irrelevance-out_of_scope-{i}-{s.scenario_id[:10]}",
            )
        )
    return items


def public_category(example: dict) -> Optional[str]:
    """BFCL category of a converted public example, or None if it fits none.

    1 tool + 1 call -> simple; 2 to 4 tools + 1 call -> multiple;
    2+ calls -> parallel; no call -> irrelevance.
    """
    calls = next((m.get("tool_calls") or [] for m in example["messages"] if m["role"] == "assistant"), [])
    n_tools = len(example.get("tools") or [])
    if not calls:
        return "irrelevance"
    if len(calls) >= 2:
        return "parallel"
    if n_tools == 1:
        return "simple"
    if 2 <= n_tools <= 4:
        return "multiple"
    return None


def _public_bfcl_items(
    rows: Iterable[dict], n: int, exclude_queries: Iterable[str]
) -> dict[str, list[EvalItem]]:
    """Up to `n` public items per BFCL category, skipping malformed rows and SFT queries."""
    excluded = set(exclude_queries)
    items: dict[str, list[EvalItem]] = {c: [] for c in BFCL_CATEGORIES}
    for index, row in enumerate(rows):
        if all(len(v) >= n for v in items.values()):
            break
        if isinstance(row, dict) and row.get("query") in excluded:
            continue
        example = convert_public_example(row)
        if isinstance(example, str):
            continue
        category = public_category(example)
        if category is None or len(items[category]) >= n:
            continue
        assistant = next(m for m in example["messages"] if m["role"] == "assistant")
        calls = [
            ToolCall(tc["function"]["name"], tc["function"]["arguments"])
            for tc in assistant.get("tool_calls") or []
        ]
        items[category].append(
            EvalItem(
                category=category,
                source="public",
                prompt_messages=[dict(m) for m in example["messages"][:2]],
                tools=example["tools"],
                gt_calls=gt_from_calls(calls),
                scenario=None,
                item_id=f"public-{category}-{index}",
            )
        )
    return items


def sft_queries(sft_rows: Iterable[dict]) -> set[str]:
    """User texts of SFT rows, to keep the public eval slice disjoint from SFT."""
    return {
        m["content"]
        for row in sft_rows
        for m in row.get("messages", [])
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    }


def build_bfcl_eval(
    sim: Optional[RelationshipSim],
    eval_scenarios: Sequence[Scenario],
    public_eval: Optional[Iterable[dict]] = None,
    n_per_category: int = N_PER_CATEGORY,
    exclude_queries: Iterable[str] = (),
) -> dict[str, list[EvalItem]]:
    """Benchmark items: {simple, multiple, parallel, irrelevance, sim} -> [EvalItem].

    Each BFCL category gets `n_per_category - n_per_category // 2` Relationship_Sim
    items (from `eval_scenarios`) and up to `n_per_category // 2` public items
    (from the raw xLAM-style rows in `public_eval`). Public rows whose `query`
    is in `exclude_queries` (e.g. `sft_queries(sft_rows)`) are skipped, so the
    public slice stays disjoint from SFT. With `public_eval=None` (public data
    failed to load) the categories hold sim items only (Req 10.14).

    The `sim` key holds one item per eval Scenario with all nine tools (empty
    `gt_calls` for No_Tool_Scenarios).
    """
    sim = sim or RelationshipSim()
    rng = random.Random(f"{sim.seed}-bfcl")
    n_public = n_per_category // 2 if public_eval is not None else 0
    n_sim = n_per_category - n_per_category // 2

    items = _sim_bfcl_items(sim, eval_scenarios, n_sim, rng)
    if n_public:
        public = _public_bfcl_items(public_eval, n_public, exclude_queries)
        for category in BFCL_CATEGORIES:
            items[category] += public[category]

    names = list(TOOLS_BY_NAME)
    items["sim"] = [
        _sim_item("sim", s, names, [] if s.is_no_tool else _flat_plan(s.optimal_plan), "all")
        for s in eval_scenarios
    ]
    return items


def check_min_per_category(
    eval_items: dict[str, Sequence[Any]], minimum: int = MIN_PER_CATEGORY
) -> None:
    """Raise if a BFCL category has fewer than `minimum` eval examples (Req 5.16).

    Raises:
        ValueError: `BFCL category 'parallel' has 12 eval examples; the minimum is 20.`
    """
    for category in BFCL_CATEGORIES:
        count = len(eval_items.get(category) or ())
        if count < minimum:
            raise ValueError(
                f"BFCL category '{category}' has {count} eval examples; the minimum is {minimum}."
            )


def print_data_summary(
    splits: dict[str, int],
    eval_items: Optional[dict[str, Sequence[EvalItem]]] = None,
    scenarios: Optional[dict[str, Sequence[Scenario]]] = None,
    dropped: Optional[dict[str, int]] = None,
    dpo_summary: Optional[dict] = None,
) -> str:
    """Print (and return) the data summary (Req 5.15, 6.4).

    Args:
        splits: examples per split, e.g. {"sft": 400, "rl": 300, "dpo": 900}.
        eval_items: `build_bfcl_eval` output; counted per category and source.
        scenarios: Scenarios per split, e.g. {"train": [...], "eval": [...]};
            counted per occasion (No_Tool shown separately).
        dropped: dropped public examples per reason (all DROP_REASONS shown).
        dpo_summary: `build_preference_pairs` summary; prints `tie_dropped`.
    """
    lines = ["Data summary", "Examples per split:"]
    lines += [f"  {name:<12} {count}" for name, count in splits.items()]

    if eval_items is not None:
        lines.append("Eval examples per category (sim + public):")
        for category in (*BFCL_CATEGORIES, "sim"):
            group = eval_items.get(category) or []
            n_sim = sum(1 for item in group if item.source == "sim")
            lines.append(f"  {category:<12} {len(group)} ({n_sim} sim + {len(group) - n_sim} public)")

    if scenarios:
        lines.append("Scenarios per occasion:")
        names = list(scenarios)
        lines.append("  " + f"{'occasion':<16}" + "".join(f"{n:>8}" for n in names))
        for occasion in OCCASIONS:
            counts = [sum(1 for s in scenarios[n] if s.state.occasion == occasion) for n in names]
            lines.append("  " + f"{occasion:<16}" + "".join(f"{c:>8}" for c in counts))
        no_tool = [sum(1 for s in scenarios[n] if s.is_no_tool) for n in names]
        lines.append("  " + f"{'(no_tool)':<16}" + "".join(f"{c:>8}" for c in no_tool))

    dropped = dict(dropped or {})
    lines.append("Dropped public examples per reason:")
    for reason in (*DROP_REASONS, *(r for r in dropped if r not in DROP_REASONS)):
        lines.append(f"  {reason:<22} {dropped.get(reason, 0)}")

    if dpo_summary is not None:
        lines.append(f"DPO pairs dropped for tied scores: {dpo_summary.get('tie_dropped', 0)}")

    text = "\n".join(lines)
    print(text)
    return text


# =============================================================================
# 16. Benchmark scoring and Reward_Hacking_Check
# =============================================================================
# The notebook generates one answer per eval item on the GPU; everything after
# that (parsing, AST scoring, simulator metrics, results rows) is plain Python
# here, so the baseline cell and the final benchmark score models the same way.
#
# Reward hacking = the reward goes up but the real task does not. Three signals:
#   high_reward_low_exact  mean reward >= 2.0 but Sim Task Success < 50%
#   format_only_gains      format + syntax rises >= 0.2 over the base model while
#                          Sim Task Success rises < 5 points
#   repeated_calls         > 10% of eval answers repeat the same tool call

#: Every key of `build_bfcl_eval` output, in scoring order.
EVAL_CATEGORIES: tuple[str, ...] = (*BFCL_CATEGORIES, "sim")

HACK_REWARD_MIN, HACK_SUCCESS_MAX = 2.0, 0.50
HACK_FORMAT_RISE_MIN, HACK_SUCCESS_RISE_MAX = 0.2, 0.05
HACK_REPEAT_SHARE_MAX = 0.10
_EPS = 1e-9  # values within float noise of a threshold count as equal to it


@dataclass(frozen=True)
class HackFlag:
    """One reward-hacking signal raised for one model."""

    model: str
    flag: str  # "high_reward_low_exact" | "format_only_gains" | "repeated_calls"
    values: dict
    explanation: str


def has_repeated_call(calls: Iterable[ToolCall]) -> bool:
    """True when the same tool call (same name and Catalog arguments) appears more than once.

    Free-text arguments are ignored, so `send_message("hi")` twice counts as a repeat.
    """
    keys = [c.key() for c in calls]
    return len(keys) != len(set(keys))


def reward_hacking_flags(model: str, stats: dict, base_stats: dict) -> list[HackFlag]:
    """Reward-hacking signals for one model (Req 12.18 to 12.20).

    `stats` / `base_stats` hold `mean_reward`, `mean_format_syntax`,
    `sim_task_success` (fractions) and `repeat_share` (fraction of eval answers
    with a repeated call). Missing (None) values never raise a flag.
    """
    flags: list[HackFlag] = []
    reward, success = stats.get("mean_reward"), stats.get("sim_task_success")
    if reward is not None and success is not None:
        if reward + _EPS >= HACK_REWARD_MIN and success + _EPS < HACK_SUCCESS_MAX:
            flags.append(HackFlag(
                model, "high_reward_low_exact",
                {"mean_reward": reward, "sim_task_success": success},
                "The reward is high but few answers match the best plan, so the model "
                "may be collecting partial reward without solving the task.",
            ))

    fmt, base_fmt = stats.get("mean_format_syntax"), base_stats.get("mean_format_syntax")
    base_success = base_stats.get("sim_task_success")
    if None not in (fmt, base_fmt, success, base_success):
        fmt_rise, success_rise = fmt - base_fmt, success - base_success
        if fmt_rise + _EPS >= HACK_FORMAT_RISE_MIN and success_rise + _EPS < HACK_SUCCESS_RISE_MAX:
            flags.append(HackFlag(
                model, "format_only_gains",
                {"format_syntax_rise": fmt_rise, "sim_task_success_rise": success_rise},
                "The model learned the answer format but not better tool choices, so "
                "most of its reward gain comes from formatting.",
            ))

    share = stats.get("repeat_share")
    if share is not None and share > HACK_REPEAT_SHARE_MAX + _EPS:
        flags.append(HackFlag(
            model, "repeated_calls",
            {"repeat_share": share},
            "Many answers repeat the same tool call, which the simulator does not "
            "reward, so the model may be padding its answers.",
        ))
    return flags


def eval_prompt_texts(items: Sequence[EvalItem], tokenizer: Any) -> list[str]:
    """Generation prompts for eval items: chat template with the item's tools.

    Every model gets the same text for the same item (Req 10.9, 12.12).
    """
    return [_render(tokenizer, item.prompt_messages, item.tools) for item in items]


def hack_stats(samples: Iterable[dict], model: Optional[str] = None) -> dict[str, Optional[float]]:
    """Reward_Hacking_Check stats from the `sim` samples (optionally of one model).

    Returns `mean_reward`, `mean_format_syntax`, `sim_task_success`, `repeat_share`
    (None when there are no sim samples).
    """
    rows = [s for s in samples if s["category"] == "sim" and (model is None or s["model"] == model)]
    return {
        "mean_reward": _mean(s["reward"] for s in rows),
        "mean_format_syntax": _mean(s["format_syntax"] for s in rows),
        "sim_task_success": accuracy(s["correct"] for s in rows),
        "repeat_share": accuracy(
            has_repeated_call(c if isinstance(c, ToolCall) else ToolCall.from_dict(c) for c in s["calls"])
            for s in rows
        ),
    }


def _score_item(item: EvalItem, text: str, sim: RelationshipSim) -> dict:
    """One sample row: parse the answer and score it for its category."""
    parse = parse_tool_calls(text)
    sample: dict[str, Any] = {
        "category": item.category, "item_id": item.item_id, "output": text,
        "think": parse.think, "calls": list(parse.calls),
        "correct": None, "love_delta": None, "reward": None, "format_syntax": None,
    }
    if item.category == "sim":
        result = sim_item_result(text, item.scenario, sim)
        names = [n for n in (_schema_name(t) for t in item.tools) if n]
        reward = score_completion(text, item.scenario, tool_names=names)
        sample.update(correct=result.success, love_delta=result.love_gain,
                      reward=reward.total, format_syntax=reward.format + reward.syntax)
    else:
        # A broken <tool_call> block counts as a wrong answer (also for irrelevance).
        sample["correct"] = not parse.errors and ast_match(parse.calls, item.gt_calls, item.category, item.tools)
    return sample


def score_eval_outputs(
    model: str,
    eval_items: dict[str, Sequence[EvalItem]],
    outputs: dict[str, Sequence[str]],
    sim: Optional[RelationshipSim] = None,
) -> dict:
    """Score one model's answers on the benchmark.

    `eval_items` is the `build_bfcl_eval` output and `outputs[category][i]` the
    model's answer to `eval_items[category][i]`. Returns a dict with
    `headline` (the four Headline_Metrics), `bfcl_per_category`, `stats` (for
    `reward_hacking_flags`), and `samples` (one row per prediction).
    """
    sim = _SIM if sim is None else sim
    samples: list[dict] = []
    per_category: dict[str, Optional[float]] = {}
    for category in EVAL_CATEGORIES:
        items, texts = list(eval_items.get(category) or ()), list(outputs.get(category) or ())
        if len(items) != len(texts):
            raise ValueError(f"{model}: {len(texts)} outputs for {len(items)} '{category}' items")
        rows = [{"model": model, **_score_item(i, t, sim)} for i, t in zip(items, texts)]
        samples += rows
        if category != "sim":
            per_category[category] = accuracy(r["correct"] for r in rows)
    sim_rows = [r for r in samples if r["category"] == "sim"]
    headline = aggregate_headline(
        per_category, [r["correct"] for r in sim_rows], [r["love_delta"] for r in sim_rows]
    )
    return {"headline": headline, "bfcl_per_category": per_category,
            "stats": hack_stats(samples, model), "samples": samples}


def benchmark_results(
    scores: dict[str, dict],
    eval_items: dict[str, Sequence[EvalItem]],
    base_model: str,
    seed: int,
    decoding: dict,
    skipped_models: Sequence[str] = (),
) -> dict:
    """Results_File / Baseline_Results dict from `score_eval_outputs` results keyed by model name."""
    return build_results_dict(
        base_model=base_model,
        seed=seed,
        decoding=decoding,
        counts={c: len(eval_items.get(c) or ()) for c in EVAL_CATEGORIES},
        headline={m: s["headline"] for m, s in scores.items()},
        bfcl_per_category={m: s["bfcl_per_category"] for m, s in scores.items()},
        skipped_models=skipped_models,
        samples=[row for s in scores.values() for row in s["samples"]],
    )


# =============================================================================
# 17. Final benchmark report
# =============================================================================
# Small helpers for the notebook's last section: reuse the baseline file for
# the base row, build the delta and before/after tables, pick example
# predictions and side-by-side Scenarios, and format reward-hacking flags.
# Scores are `score_eval_outputs` results keyed by model name ("base", "sft", ...).

#: Which eval categories feed each Headline_Metric (for the example predictions).
METRIC_CATEGORIES: dict[str, tuple[str, ...]] = {
    "bfcl_ast_acc": ("simple", "multiple", "parallel"),
    "irrelevance_acc": ("irrelevance",),
    "sim_task_success": ("sim",),
    "mean_love_gain": ("sim",),
}
DELTA = "\u0394"  # "Δ" column of the before/after table


def _as_calls(calls: Iterable[Any]) -> list[ToolCall]:
    """ToolCalls from a sample's `calls` (ToolCall objects or JSON dicts)."""
    return [c if isinstance(c, ToolCall) else ToolCall.from_dict(c) for c in calls or ()]


def load_baseline_scores(
    path: Union[str, os.PathLike],
    eval_items: dict[str, Sequence[EvalItem]],
    base_model: str,
    seed: int,
    decoding: dict,
) -> Optional[dict]:
    """The base model's scores from the Baseline_Results file (Req 12.14).

    Returns a `score_eval_outputs`-style dict, or None (score the base model
    again) when the file is missing or was made with another base model, seed,
    decoding, or eval set.
    """
    p = Path(path)
    if not p.is_file():
        print(f"No baseline file at {p}; the base model is scored again.")
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    counts = {c: len(eval_items.get(c) or ()) for c in EVAL_CATEGORIES}
    expected = (base_model, seed, _jsonable(dict(decoding)), counts)
    found = (data.get("base_model"), data.get("seed"), data.get("decoding"), data.get("counts"))
    if found != expected or "base" not in (data.get("headline") or {}):
        print(f"{p} does not match the current base model, seed, decoding, or eval set; "
              "the base model is scored again.")
        return None
    samples = [dict(s, calls=_as_calls(s.get("calls"))) for s in data.get("samples", []) if s.get("model") == "base"]
    print(f"Base row reused from {p}")
    return {
        "headline": data["headline"]["base"],
        "bfcl_per_category": (data.get("bfcl_per_category") or {}).get("base", {}),
        "stats": hack_stats(samples, "base"),
        "samples": samples,
    }


def delta_rows(scores: dict[str, dict], base: str = "base") -> dict[str, dict[str, str]]:
    """Signed change of each Headline_Metric vs the base model, one row per trained model."""
    ref = scores[base]["headline"]
    return {m: format_delta_row(headline_deltas(s["headline"], ref)) for m, s in scores.items() if m != base}


def before_after_rows(scores: dict[str, dict], base: str = "base") -> dict[str, dict[tuple[str, str], str]]:
    """Before/after table: per trained model and metric, `base | trained | Δ` (Req 12.16).

    Keys are (metric label, column) tuples, so `pd.DataFrame.from_dict(rows,
    orient="index")` gives two-level column headers.
    """
    ref = scores[base]["headline"]
    base_row = format_headline_row(ref)
    rows: dict[str, dict[tuple[str, str], str]] = {}
    for m, s in scores.items():
        if m == base:
            continue
        cols = (("base", base_row), ("trained", format_headline_row(s["headline"])),
                (DELTA, format_delta_row(headline_deltas(s["headline"], ref))))
        rows[m] = {(label, col): row[label] for label in HEADLINE_COLUMNS.values() for col, row in cols}
    return rows


def best_trained_model(scores: dict[str, dict], base: str = "base") -> Optional[str]:
    """Trained model with the highest `Sim Task Success` (first one on ties; None if none scored)."""
    trained = [m for m in scores if m != base]
    if not trained:
        return None
    value = lambda m: scores[m]["headline"].get("sim_task_success")
    return max(trained, key=lambda m: (value(m) is not None, value(m) or 0.0))


def _user_text(item: EvalItem) -> str:
    """The last user message of an eval item (the request the model answers)."""
    return next((m["content"] for m in reversed(item.prompt_messages) if m.get("role") == "user"), "")


def _short(text: Optional[str], limit: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _calls_text(calls: Iterable[Any]) -> str:
    return ", ".join(repr(c) for c in _as_calls(calls)) or "(no tool call)"


def metric_examples(samples: Iterable[dict], eval_items: dict[str, Sequence[EvalItem]], model: str, n: int = 2) -> str:
    """`n` example predictions per Headline_Metric for one model (Req 10.15).

    Each example shows the prompt (user request), think text, tool calls and
    score. Examples are the first items of the metric's categories (taken in
    turn, so BFCL AST Acc shows a `simple` and a `multiple` item); metrics that
    share the `sim` category show different items. Every model gets the same items.
    """
    items = {(c, it.item_id): it for c in EVAL_CATEGORIES for it in eval_items.get(c) or ()}
    rows = [s for s in samples if s["model"] == model]
    used: set[tuple[str, str]] = set()
    lines = [f"===== Example predictions: {model} ====="]
    for metric, categories in METRIC_CATEGORIES.items():
        pools = [[s for s in rows if s["category"] == c and (c, s["item_id"]) not in used] for c in categories]
        picked = [pool[i] for i in range(n) for pool in pools if i < len(pool)][:n]
        for s in picked:
            used.add((s["category"], s["item_id"]))
            item = items.get((s["category"], s["item_id"]))
            if metric == "mean_love_gain":
                score = f"Love_Delta {s['love_delta']:+d}, reward {s['reward']:.2f}"
            else:
                score = "correct" if s["correct"] else "wrong"
            lines += [
                f"[{HEADLINE_COLUMNS[metric]}] {s['item_id']}",
                f"  prompt:     {_short(_user_text(item)) if item else '(item not found)'}",
                f"  think:      {_short(s.get('think')) or '(none)'}",
                f"  tool calls: {_calls_text(s.get('calls'))}",
                f"  score:      {score}",
            ]
        if not picked:
            lines.append(f"[{HEADLINE_COLUMNS[metric]}] no examples ({NA})")
    return "\n".join(lines)


def side_by_side_items(eval_items: dict[str, Sequence[EvalItem]], n: int = 3) -> list[EvalItem]:
    """Fixed `sim` items to compare models: the first tool Scenario, the first
    `after_argument` tool Scenario, the first No_Tool_Scenario; then further
    items in order until there are `n`."""
    sims = [it for it in eval_items.get("sim") or () if it.scenario is not None]
    tests = (
        lambda s: not s.is_no_tool,
        lambda s: not s.is_no_tool and s.state.occasion == "after_argument",
        lambda s: s.is_no_tool,
    )
    picked: list[EvalItem] = []
    for test in (*tests, *([lambda s: True] * n)):
        item = next((it for it in sims if all(it is not p for p in picked) and test(it.scenario)), None)
        if item is not None and len(picked) < n:
            picked.append(item)
    return picked


def side_by_side_rows(
    scores: dict[str, dict], eval_items: dict[str, Sequence[EvalItem]], model: str, base: str = "base", n: int = 3
) -> list[dict[str, str]]:
    """One row per side-by-side Scenario: request, then calls / Love_Delta / reward
    for the base model and for `model` (Req 12.17)."""
    by_id = {m: {s["item_id"]: s for s in scores[m]["samples"] if s["category"] == "sim"} for m in (base, model)}
    rows = []
    for item in side_by_side_items(eval_items, n):
        row = {"occasion": item.scenario.state.occasion, "request": item.scenario.request}
        for m in (base, model):
            s = by_id[m].get(item.item_id)
            row[f"{m} calls"] = _calls_text(s["calls"]) if s else NA
            row[f"{m} Love_Delta"] = f"{s['love_delta']:+d}" if s else NA
            row[f"{m} reward"] = f"{s['reward']:.2f}" if s else NA
        rows.append(row)
    return rows


def format_hack_flag(flag: HackFlag) -> str:
    """'grpo: high_reward_low_exact (mean_reward=2.100, sim_task_success=0.400). <explanation>'"""
    values = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in flag.values.items())
    return f"{flag.model}: {flag.flag} ({values}). {flag.explanation}"
