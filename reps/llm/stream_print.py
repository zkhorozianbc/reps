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
