"""Tests for reps.sanitize — the conclusion-hedging primitive.

Covers:
  * detector behavior (false positives, idiomatic phrases)
  * deterministic regex_sanitizer (idempotent, schema-preserving)
  * make_llm_sanitizer (success path, exception path, residual-marker
    fallback, clean-input short-circuit)
  * sanitize_schema (str/list/dict/tuple/scalar walking, structure
    preservation)
"""
from __future__ import annotations

import pytest

from reps.sanitize import (
    LLM_REWRITE_SYSTEM_PROMPT,
    has_absolute_claim,
    make_llm_sanitizer,
    regex_sanitizer,
    sanitize_schema,
)


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "improvement is mathematically impossible",
    "this approach cannot be improved",
    "we have exhausted all viable strategies",
    "reached the theoretical limit",
    "provably optimal under L2",
    "no further improvement is possible at this score",
    "the optimal solution found uses BFS",
    "It's impossible to do better here.",
])
def test_has_absolute_claim_positive(text):
    assert has_absolute_claim(text)


@pytest.mark.parametrize("text", [
    "",
    "this approach is hard to improve",
    "we tried sorting and it plateaued",
    "no clear next direction",
    "swap heap for treap",
    "improvement was difficult in this batch",
    "score 0.42; parent 0.41",
])
def test_has_absolute_claim_negative(text):
    assert not has_absolute_claim(text)


# ---------------------------------------------------------------------------
# regex_sanitizer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regex_sanitizer_rewrites_canonical_failure_phrase():
    text = "improvement is mathematically impossible"
    out = await regex_sanitizer(text)
    assert not has_absolute_claim(out)
    assert "appears intractable" in out


@pytest.mark.asyncio
async def test_regex_sanitizer_idempotent_on_clean_input():
    text = "swap quicksort for mergesort; expected delta +0.01"
    out = await regex_sanitizer(text)
    assert out == text


@pytest.mark.asyncio
async def test_regex_sanitizer_idempotent_on_already_hedged():
    text = "improvement appears intractable so far in this batch"
    out = await regex_sanitizer(text)
    out2 = await regex_sanitizer(out)
    assert out == out2


@pytest.mark.asyncio
async def test_regex_sanitizer_handles_multiple_markers():
    text = "improvement is impossible and we have exhausted all approaches"
    out = await regex_sanitizer(text)
    assert not has_absolute_claim(out)


# ---------------------------------------------------------------------------
# make_llm_sanitizer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_sanitizer_skips_clean_input():
    """Cheap path: clean input must not invoke the LLM."""
    calls = []

    async def rewrite(text: str) -> str:
        calls.append(text)
        return "should not be called"

    sanitizer = make_llm_sanitizer(rewrite)
    out = await sanitizer("swap heap for treap")
    assert out == "swap heap for treap"
    assert calls == []


@pytest.mark.asyncio
async def test_llm_sanitizer_uses_rewrite_on_marked_input():
    async def rewrite(text: str) -> str:
        return "improvement appears difficult based on this batch"

    sanitizer = make_llm_sanitizer(rewrite)
    out = await sanitizer("improvement is mathematically impossible")
    assert not has_absolute_claim(out)
    assert "appears difficult" in out


@pytest.mark.asyncio
async def test_llm_sanitizer_falls_back_when_rewrite_raises():
    async def rewrite(text: str) -> str:
        raise RuntimeError("API timeout")

    sanitizer = make_llm_sanitizer(rewrite)
    out = await sanitizer("improvement is mathematically impossible")
    assert not has_absolute_claim(out)


@pytest.mark.asyncio
async def test_llm_sanitizer_falls_back_when_rewrite_leaves_marker():
    """If the LLM rewrite still contains a marker, regex must clean up."""
    async def rewrite(text: str) -> str:
        return "this is still impossible somehow"

    sanitizer = make_llm_sanitizer(rewrite)
    out = await sanitizer("improvement is mathematically impossible")
    assert not has_absolute_claim(out)


# ---------------------------------------------------------------------------
# sanitize_schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sanitize_schema_walks_dict_of_lists():
    """Reflection-engine 4-field shape — strings deep inside dict-of-lists."""
    payload = {
        "working_patterns": ["mergesort beats quicksort here"],
        "failing_patterns": ["randomization is mathematically impossible"],
        "hypotheses": ["score plateaus near 0.42"],
        "suggested_directions": ["try a heap-based pivot"],
    }
    out = await sanitize_schema(payload, regex_sanitizer)
    # shape preserved
    assert set(out.keys()) == set(payload.keys())
    assert all(isinstance(v, list) for v in out.values())
    # marked field rewritten
    assert not has_absolute_claim(out["failing_patterns"][0])
    # clean fields untouched
    assert out["working_patterns"] == payload["working_patterns"]
    assert out["hypotheses"] == payload["hypotheses"]


@pytest.mark.asyncio
async def test_sanitize_schema_preserves_summarizer_shape():
    """Program-summarizer 3-field shape — list of pitfalls + scalar fields."""
    payload = {
        "approach": "tried impossible-to-prove invariant",
        "pitfalls": ["division by zero", "no improvement is possible"],
        "key_insight": "sort key needs tie-breaker",
    }
    out = await sanitize_schema(payload, regex_sanitizer)
    assert set(out.keys()) == {"approach", "pitfalls", "key_insight"}
    assert len(out["pitfalls"]) == 2
    assert all(not has_absolute_claim(s) for s in out["pitfalls"])
    assert not has_absolute_claim(out["approach"])


@pytest.mark.asyncio
async def test_sanitize_schema_handles_mixed_scalars():
    """Non-string scalars must pass through unchanged."""
    payload = {"score": 0.42, "improved": True, "note": "impossible to do better", "n": None}
    out = await sanitize_schema(payload, regex_sanitizer)
    assert out["score"] == 0.42
    assert out["improved"] is True
    assert out["n"] is None
    assert not has_absolute_claim(out["note"])


@pytest.mark.asyncio
async def test_sanitize_schema_does_not_mutate_input():
    payload = {"failing_patterns": ["mathematically impossible"]}
    snapshot = {"failing_patterns": ["mathematically impossible"]}
    _ = await sanitize_schema(payload, regex_sanitizer)
    assert payload == snapshot


@pytest.mark.asyncio
async def test_sanitize_schema_preserves_tuple_type():
    payload = ("clean text", "improvement is impossible")
    out = await sanitize_schema(payload, regex_sanitizer)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0] == "clean text"
    assert not has_absolute_claim(out[1])


# ---------------------------------------------------------------------------
# module surface
# ---------------------------------------------------------------------------


def test_llm_rewrite_prompt_is_exposed():
    assert isinstance(LLM_REWRITE_SYSTEM_PROMPT, str)
    assert len(LLM_REWRITE_SYSTEM_PROMPT) > 0
