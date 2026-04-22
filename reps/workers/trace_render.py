"""Human-readable pretty-printer for List[TurnRecord]. Used by CLI inspection
and post-hoc log viewing."""
from __future__ import annotations

import json
from typing import List

from reps.workers.base import TurnRecord


def render_trace(turns: List[TurnRecord]) -> str:
    out: list[str] = []
    for t in turns:
        header = f"=== turn {t.index} [{t.role}]"
        if t.model_id:
            header += f" {t.model_id}"
        if t.stop_reason:
            header += f" stop={t.stop_reason}"
        out.append(header + " ===")
        for b in t.blocks:
            if b.type == "text":
                out.append(b.text or "")
            elif b.type == "thinking":
                sig = "[signed]" if b.signature else "[unsigned]"
                out.append(f"[thinking {sig}]\n{b.text or ''}")
            elif b.type == "redacted_thinking":
                out.append("[thinking REDACTED]")
            elif b.type == "tool_use":
                inp = json.dumps(b.tool_input, indent=2) if b.tool_input is not None else ""
                out.append(f"[tool_use {b.tool_name} id={b.tool_use_id}]\n  input: {inp}")
            elif b.type == "tool_result":
                err = " ERROR" if b.tool_result_is_error else ""
                body = (
                    b.tool_result_content
                    if isinstance(b.tool_result_content, str)
                    else json.dumps(b.tool_result_content, indent=2)
                )
                out.append(f"[tool_result for={b.tool_result_for_id}{err}]\n{body}")
        if t.usage:
            out.append(
                f"  usage: in={t.usage.get('input_tokens')} "
                f"out={t.usage.get('output_tokens')} "
                f"cache_read={t.usage.get('cache_read_input_tokens', 0)}"
            )
        out.append("")
    return "\n".join(out)
