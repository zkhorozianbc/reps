"""Serialize a sequence of (search, replace) diff blocks into canonical REPS
SEARCH/REPLACE text format.

Output is the inverse of reps.utils.extract_diffs — feed it back through
extract_diffs and you get the input blocks.

Rule (from spec §5 and applied_edit research):
  - `applied_edit` for tool-calling workers in diff-mode is this serialization
    of all edit_file blocks in APPLY ORDER (effort-preserving — overrides and
    re-edits are kept, not deduped).
  - ConvergenceMonitor.classify_edit only checks length + keyword presence
    (reps/convergence_monitor.py:30-70), so a multi-block concatenation gives
    a sensible edit-entropy signal.
"""
from __future__ import annotations

from typing import Iterable, Tuple


def serialize_diff_blocks(blocks: Iterable[Tuple[str, str]]) -> str:
    """Return canonical REPS SEARCH/REPLACE text for the given blocks."""
    parts: list[str] = []
    for search, replace in blocks:
        parts.append("<<<<<<< SEARCH\n")
        parts.append(search)
        if not search.endswith("\n"):
            parts.append("\n")
        parts.append("=======\n")
        parts.append(replace)
        if not replace.endswith("\n"):
            parts.append("\n")
        parts.append(">>>>>>> REPLACE\n")
    return "".join(parts)
