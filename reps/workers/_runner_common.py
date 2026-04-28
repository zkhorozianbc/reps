"""Helpers shared between the Anthropic and OpenAI tool-runner workers.

Extracted from reps/workers/anthropic_tool_runner.py and openai_tool_runner.py
to remove copy-paste of identical small utilities (full-rewrite-tail stripping,
placeholder rejection, applied-edit diff). Provider-specific request shapes
remain in their respective files; this module only owns logic that is
provider-agnostic.
"""
from __future__ import annotations

import difflib
from typing import Optional, Tuple


PLACEHOLDER_TOKENS = frozenset(
    {
        "code_placeholder",
        "PLACEHOLDER_USE_INFLIGHT",
        "TODO",
        "...",
    }
)
_MIN_SUBMITTED_CHARS = 30


def strip_full_rewrite_tail(user_text: str) -> str:
    """Remove the full-rewrite task framing that the sampler's
    full_rewrite_user.txt template appends.

    The tool-runner uses edit_file mechanics; a "Provide the complete new
    program code" instruction plus a ```python # Your rewritten program here ```
    skeleton pushes the model toward full-rewrite dumps instead of targeted
    edits. Strategy: locate the "# Task" header that precedes that block and
    cut from there to the end. Falls back to locating the skeleton itself.
    """
    if not user_text:
        return user_text
    idx = user_text.find("\n# Task\n")
    if idx == -1:
        skel = user_text.find("# Your rewritten program here")
        if skel == -1:
            return user_text
        fence = user_text.rfind("```", 0, skel)
        if fence != -1:
            return user_text[:fence].rstrip() + "\n"
        return user_text[:skel].rstrip() + "\n"
    return user_text[:idx].rstrip() + "\n"


def reject_placeholder_submission(code: Optional[str]) -> Optional[str]:
    """Return an error string if the submitted code is empty / placeholder /
    suspiciously short, else None. Both tool-runners use this to reject
    submit_child payloads that would clobber the parent with sentinel strings.
    """
    stripped = (code or "").strip()
    if (
        not stripped
        or stripped in PLACEHOLDER_TOKENS
        or len(stripped) < _MIN_SUBMITTED_CHARS
    ):
        return (
            f"REJECTED: submitted code is empty / placeholder / "
            f"too short (len={len(stripped)}). Retry with actual "
            f"program code."
        )
    return None


def compute_applied_edit(
    code: str,
    parent_code: str,
    parent_id: str,
    generation_mode: str,
) -> str:
    """Return either the full child code (full-rewrite mode) or a unified diff
    against the parent (diff mode). Used as the persisted "applied edit" for
    metrics and prompt construction."""
    if generation_mode == "full":
        return code
    parent_lines = parent_code.splitlines(keepends=True)
    child_lines = code.splitlines(keepends=True)
    diff = "".join(
        difflib.unified_diff(
            parent_lines,
            child_lines,
            fromfile=f"parent/{parent_id}",
            tofile="child/new",
            n=3,
        )
    )
    return diff or "# no textual change"
