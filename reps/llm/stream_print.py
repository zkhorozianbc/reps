"""Pretty-print a completed LLM output block to stderr.

Shared between the OpenRouter/OpenAI streaming path and the Anthropic
native streaming path so both providers surface reasoning and answer
blocks the same way. Keeps it simple — no rich library, no colors.
Each block: one header line with pid + kind, the block text, a blank
line separator. Parallel workers interleave at block granularity (one
complete block at a time) rather than chunk-by-chunk chaos.
"""

import os
import sys
from typing import Optional


def emit_block(kind: str, text: Optional[str]) -> None:
    """Print a completed block (thinking or answer) to stderr as a unit.

    No-op when text is empty/whitespace.
    """
    if not text:
        return
    body = text.rstrip()
    if not body:
        return
    pid = os.getpid()
    header = f"━━━ [pid={pid} {kind}] ━━━"
    sys.stderr.write(f"\n{header}\n{body}\n")
    sys.stderr.flush()


def emit_status(model: str, kind: str = "stream opened") -> None:
    """Print a one-line status marker so the user sees the worker is alive
    before the first content chunk arrives. Reasoning models often buffer
    reasoning upstream for tens of seconds before streaming anything."""
    pid = os.getpid()
    sys.stderr.write(f"[pid={pid}] {kind}: {model}\n")
    sys.stderr.flush()
