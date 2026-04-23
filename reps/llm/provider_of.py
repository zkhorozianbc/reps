"""Tiny utility: map a model id to its upstream provider name.

Centralizes the "claude-* → anthropic, gpt-*/o*-* → openai" heuristic so
callers that build a standalone LLM client (e.g. the per-program summarizer)
can pick the right provider class without duplicating string tests.

Kept deliberately small: the single source of truth is this mapping + the
explicit `provider` override on the caller's config. If the heuristic can't
decide, we raise — silent fallback to a default provider is a footgun that
masks real config mistakes.
"""

from __future__ import annotations


def provider_of_model(name: str) -> str:
    """Return "anthropic" or "openai" based on the model id.

    Recognized markers (case-insensitive on the final path segment):
      - anthropic:   claude-*, anthropic/*
      - openai:      gpt-*, o1-*, o3-*, o4-*, openai/*

    Raises ValueError if the name matches neither family. Callers that
    need a non-heuristic path (e.g. third-party / OpenRouter-hosted models)
    should set `provider` explicitly on their config and skip this helper.
    """
    if not name:
        raise ValueError("provider_of_model: empty model name")
    raw = name.strip()
    lowered = raw.lower()

    # Explicit "provider/model" prefixes win immediately.
    if lowered.startswith("anthropic/"):
        return "anthropic"
    if lowered.startswith("openai/"):
        return "openai"

    # Use the final path segment for the family check so "anything/claude-x"
    # still resolves correctly when someone puts a custom prefix in front.
    base = lowered.rsplit("/", 1)[-1]

    if base.startswith("claude-") or base.startswith("claude"):
        return "anthropic"
    # OpenAI reasoning + chat model families.
    if base.startswith(("gpt-", "gpt", "o1-", "o1", "o3-", "o3", "o4-", "o4")):
        return "openai"

    raise ValueError(
        f"provider_of_model: cannot infer provider from {name!r}. "
        f"Set `provider` explicitly on the config."
    )
