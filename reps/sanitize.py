"""Hedge absolute-pessimism in LLM-generated prose.

A small portable primitive. Any producer that emits prose carrying
conclusions (batch reflection, per-program summary, future producers)
can compose this transform inline without changing its schema:

    data = await sanitize_schema(parsed_json, sanitizer)

Composition contract:

    Sanitizer = Callable[[str], Awaitable[str]]

Producers store one Sanitizer; the controller chooses which implementation
to wire in. Two are provided here:

  * ``regex_sanitizer`` — pure, deterministic, free. Idempotent on clean
    input. Always available as a fallback.
  * ``make_llm_sanitizer(rewrite)`` — wraps an async LLM rewrite callable.
    Cheap path: regex-screens first; the LLM is only invoked when an
    absolute-pessimism marker is present. If the rewrite raises or fails
    to remove markers, falls back to the regex sanitizer.

The schema walker ``sanitize_schema`` applies a Sanitizer to every string
leaf of a JSON-shaped object, preserving structure. Producers do not need
to enumerate their own fields.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

Sanitizer = Callable[[str], Awaitable[str]]
T = TypeVar("T")


# Markers that signal an absolute-pessimism claim. Matched case-insensitively.
# Conservative on purpose: phrases that assert proven impossibility, not mere
# difficulty. A trace that says "this is hard" is not defeatist; one that says
# "this is mathematically impossible" is.
_ABSOLUTE_MARKERS = re.compile(
    r"\b("
    r"impossible"
    r"|cannot\s+be\s+(?:improved|surpassed|beaten|optimized)"
    r"|mathematical(?:ly)?\s+(?:impossible|optimal|proven|limit)"
    r"|provably\s+optimal"
    r"|no\s+(?:further\s+|more\s+)?improvement(?:s)?\s+(?:is|are)\s+possible"
    r"|exhausted\s+(?:all|every)"
    r"|reached\s+(?:the\s+)?(?:theoretical\s+)?(?:limit|maximum|optimum|ceiling)"
    r"|optimal\s+solution\s+(?:found|reached|achieved)"
    r")\b",
    re.IGNORECASE,
)


# Deterministic substitutions for the regex-only fallback. Order matters:
# longer, more specific phrases must run before single-word fallbacks so
# that "mathematically impossible" is rewritten before "impossible" alone.
_HEDGE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmathematical(?:ly)?\s+impossible\b", re.IGNORECASE),
     "appears intractable so far"),
    (re.compile(r"\bmathematical(?:ly)?\s+(?:optimal|proven|limit)\b", re.IGNORECASE),
     "best observed so far"),
    (re.compile(r"\bprovably\s+optimal\b", re.IGNORECASE),
     "best observed so far"),
    (re.compile(r"\bcannot\s+be\s+(improved|surpassed|beaten|optimized)\b", re.IGNORECASE),
     r"has not been \1 in this batch"),
    (re.compile(r"\bno\s+(?:further\s+|more\s+)?improvement(?:s)?\s+(?:is|are)\s+possible\b", re.IGNORECASE),
     "no improvement was found in this batch"),
    (re.compile(r"\bexhausted\s+(all|every)\b", re.IGNORECASE),
     r"have not yet covered \1"),
    (re.compile(r"\breached\s+(?:the\s+)?(?:theoretical\s+)?(?:limit|maximum|optimum|ceiling)\b", re.IGNORECASE),
     "approaching the best score observed"),
    (re.compile(r"\boptimal\s+solution\s+(found|reached|achieved)\b", re.IGNORECASE),
     r"strong candidate \1"),
    (re.compile(r"\bimpossible\s+to\s+(\w+)\b", re.IGNORECASE),
     r"difficult to \1"),
    (re.compile(r"\bimpossible\b", re.IGNORECASE),
     "difficult so far"),
]


# Default system prompt for the LLM-backed sanitizer. Exposed as a module
# constant so callers that build their own rewrite wrapper (e.g. with a
# different provider plumbing) can reuse it.
LLM_REWRITE_SYSTEM_PROMPT = (
    "You rewrite text to remove unjustified certainty about impossibility. "
    "Replace claims of proven optimality, mathematical impossibility, or "
    "exhaustion (e.g. 'mathematically impossible', 'cannot be improved', "
    "'provably optimal', 'exhausted everything') with hedged forms that "
    "acknowledge what was observed without asserting absolutes (e.g. "
    "'appears difficult based on this batch', 'has not been improved in "
    "this batch', 'best observed so far'). Preserve all other content "
    "verbatim — same numbers, same code references, same structure, same "
    "list of items. Output ONLY the rewritten text. No preamble, no "
    "commentary, no markdown fences."
)


def has_absolute_claim(text: str) -> bool:
    """Pure detector. True iff `text` contains an absolute-pessimism marker."""
    return bool(text) and bool(_ABSOLUTE_MARKERS.search(text))


def _regex_hedge(text: str) -> str:
    if not has_absolute_claim(text):
        return text
    out = text
    for pattern, replacement in _HEDGE_RULES:
        out = pattern.sub(replacement, out)
    return out


async def regex_sanitizer(text: str) -> str:
    """Deterministic hedge. Zero cost. Idempotent on clean input."""
    return _regex_hedge(text)


def make_llm_sanitizer(
    rewrite: Callable[[str], Awaitable[str]],
    *,
    fallback: Sanitizer = regex_sanitizer,
) -> Sanitizer:
    """Wrap an async LLM rewrite callable into a Sanitizer.

    The rewrite callable is invoked only when the input contains an absolute
    marker, so clean inputs cost nothing. On rewrite exception, or if the
    rewrite did not remove all markers, falls back to ``fallback``
    (``regex_sanitizer`` by default) so the sanitizer never raises and
    never returns marked text.
    """
    async def sanitizer(text: str) -> str:
        if not has_absolute_claim(text):
            return text
        try:
            rewritten = await rewrite(text)
        except Exception as e:
            logger.warning("llm sanitizer rewrite failed; falling back: %s", e)
            return await fallback(text)
        if has_absolute_claim(rewritten):
            return await fallback(rewritten)
        return rewritten
    return sanitizer


async def sanitize_schema(obj: T, sanitizer: Sanitizer = regex_sanitizer) -> T:
    """Walk a JSON-shaped object and apply ``sanitizer`` to every string leaf.

    Preserves structure exactly: dicts keep their keys, lists keep their
    length and order, tuples stay tuples, non-string scalars pass through.
    Returns a new object; does not mutate the input.
    """
    if isinstance(obj, str):
        return await sanitizer(obj)  # type: ignore[return-value]
    if isinstance(obj, list):
        return [await sanitize_schema(x, sanitizer) for x in obj]  # type: ignore[return-value]
    if isinstance(obj, tuple):
        items = [await sanitize_schema(x, sanitizer) for x in obj]
        return tuple(items)  # type: ignore[return-value]
    if isinstance(obj, dict):
        return {k: await sanitize_schema(v, sanitizer) for k, v in obj.items()}  # type: ignore[return-value]
    return obj
