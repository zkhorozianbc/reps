"""Human-readable pretty-printer for List[TurnRecord]. Used by CLI inspection
and post-hoc log viewing."""
from __future__ import annotations

import json
from typing import Any, Dict, List

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


def render_trace_from_dicts(turns: List[Dict[str, Any]]) -> str:
    """Same output as render_trace, but input is the dict form stored in
    program.metadata['turns']."""
    out: list[str] = []
    for t in turns:
        index = t.get("index", 0)
        role = t.get("role", "unknown")
        model_id = t.get("model_id") or ""
        stop_reason = t.get("stop_reason") or ""
        usage = t.get("usage") or {}

        header = f"=== turn {index} [{role}]"
        if model_id:
            header += f" {model_id}"
        if stop_reason:
            header += f" stop={stop_reason}"
        out.append(header + " ===")

        for b in t.get("blocks", []) or []:
            btype = b.get("type") if isinstance(b, dict) else None
            if btype == "text":
                out.append(b.get("text") or "")
            elif btype == "thinking":
                sig = "[signed]" if b.get("signature") else "[unsigned]"
                out.append(f"[thinking {sig}]\n{b.get('text') or ''}")
            elif btype == "redacted_thinking":
                out.append("[thinking REDACTED]")
            elif btype == "tool_use":
                tool_input = b.get("tool_input")
                inp = json.dumps(tool_input, indent=2) if tool_input is not None else ""
                out.append(
                    f"[tool_use {b.get('tool_name')} id={b.get('tool_use_id')}]\n  input: {inp}"
                )
            elif btype == "tool_result":
                err = " ERROR" if b.get("tool_result_is_error") else ""
                body = b.get("tool_result_content", "")
                if not isinstance(body, str):
                    body = json.dumps(body, indent=2)
                out.append(
                    f"[tool_result for={b.get('tool_result_for_id')}{err}]\n{body}"
                )

        if usage:
            out.append(
                f"  usage: in={usage.get('input_tokens')} "
                f"out={usage.get('output_tokens')} "
                f"cache_read={usage.get('cache_read_input_tokens', 0)}"
            )
        out.append("")
    return "\n".join(out)
