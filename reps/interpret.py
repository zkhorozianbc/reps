"""Interpretation functions for multi-instance evaluation.

An `Interpretation` collapses a distribution of per-instance scores into the
scalar that drives selection, promotion, and best-program tracking. Pareto,
trace reflection, and merge continue to read the full per-instance vector
directly — the interpretation only governs the scalar axis.

Why this exists:
    `evaluate(code) -> float` discards information; even when an evaluator
    populates `EvaluationResult.per_instance_scores`, the rest of the harness
    used to read `metrics["combined_score"]` blindly. That coupled the user's
    measurement to a single hardcoded reduction. With an interpretation, the
    user separates "what we measured per instance" from "how we promote",
    and can swap mean / worst-case / quantile / CVaR / pass-rate without
    rewriting the evaluator.

Contract:
    Interpretation = Callable[
        [Optional[Mapping[str, float]], Mapping[str, float]],
        float,
    ]

    Args:
        per_instance: per-instance score dict (may be None or empty).
        metrics:      the metrics dict from `EvaluationResult.metrics`.
                      Provided as a fallback information source — `combined()`
                      reads `metrics["combined_score"]` first.

    Returns: float scalar used as `combined_score` for selection.

All built-in interpretations are factory functions returning a callable, so
parameterized (`cvar(0.1)`) and unparameterized (`mean()`) call sites look
uniform.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Optional

Interpretation = Callable[[Optional[Mapping[str, float]], Mapping[str, float]], float]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _finite_values(per_instance: Optional[Mapping[str, float]]) -> list[float]:
    """Drop None / NaN / inf so a single broken eval doesn't poison the scalar.

    Empty input (None or {}) returns []. Callers fall back to metrics["combined_score"]
    when they get back an empty list.
    """
    if not per_instance:
        return []
    out: list[float] = []
    for v in per_instance.values():
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _fallback(metrics: Mapping[str, float], default: float = 0.0) -> float:
    """Return metrics['combined_score'] if present and finite, else `default`."""
    raw = metrics.get("combined_score") if metrics else None
    if raw is None:
        return float(default)
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return f if math.isfinite(f) else float(default)


# ---------------------------------------------------------------------------
# built-in interpretations
# ---------------------------------------------------------------------------


def combined(fallback: Optional[Interpretation] = None) -> Interpretation:
    """Default. Read `metrics['combined_score']` if present, else delegate.

    Preserves legacy behavior: benchmarks that already publish a
    `combined_score` key keep working unchanged. When that key is absent
    (or non-finite), falls through to `fallback` (default: `mean()`).
    """
    fb = fallback if fallback is not None else mean()

    def _combined(per_instance: Optional[Mapping[str, float]],
                  metrics: Mapping[str, float]) -> float:
        raw = metrics.get("combined_score") if metrics else None
        if raw is not None:
            try:
                f = float(raw)
            except (TypeError, ValueError):
                f = float("nan")
            if math.isfinite(f):
                return f
        return fb(per_instance, metrics)

    return _combined


def mean() -> Interpretation:
    """Arithmetic mean of finite per-instance scores. Risk-neutral."""
    def _mean(per_instance: Optional[Mapping[str, float]],
              metrics: Mapping[str, float]) -> float:
        vals = _finite_values(per_instance)
        if not vals:
            return _fallback(metrics)
        return sum(vals) / len(vals)
    return _mean


def worst() -> Interpretation:
    """Min over finite per-instance scores. Robust / pessimistic."""
    def _worst(per_instance: Optional[Mapping[str, float]],
               metrics: Mapping[str, float]) -> float:
        vals = _finite_values(per_instance)
        if not vals:
            return _fallback(metrics)
        return min(vals)
    return _worst


def best() -> Interpretation:
    """Max over finite per-instance scores. Optimistic / upper-bound."""
    def _best(per_instance: Optional[Mapping[str, float]],
              metrics: Mapping[str, float]) -> float:
        vals = _finite_values(per_instance)
        if not vals:
            return _fallback(metrics)
        return max(vals)
    return _best


def quantile(q: float) -> Interpretation:
    """q-th quantile via linear interpolation. q in [0, 1]."""
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile q must be in [0, 1], got {q!r}")

    def _quantile(per_instance: Optional[Mapping[str, float]],
                  metrics: Mapping[str, float]) -> float:
        vals = sorted(_finite_values(per_instance))
        if not vals:
            return _fallback(metrics)
        if len(vals) == 1:
            return vals[0]
        # Linear interpolation between adjacent ranks (numpy "linear" method).
        pos = q * (len(vals) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        frac = pos - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac
    return _quantile


def cvar(alpha: float) -> Interpretation:
    """Conditional value at risk: mean of the worst alpha fraction.

    `alpha=0.1` → "average score over the worst 10% of instances."
    Equivalent to `mean()` when alpha == 1, `worst()` in the limit alpha → 0.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"cvar alpha must be in (0, 1], got {alpha!r}")

    def _cvar(per_instance: Optional[Mapping[str, float]],
              metrics: Mapping[str, float]) -> float:
        vals = sorted(_finite_values(per_instance))
        if not vals:
            return _fallback(metrics)
        # ceil so alpha=0.1 over 5 instances picks 1 (not 0); never zero-length.
        k = max(1, math.ceil(alpha * len(vals)))
        tail = vals[:k]
        return sum(tail) / len(tail)
    return _cvar


def weighted(weights: Mapping[str, float]) -> Interpretation:
    """Weighted average: sum(w_i * score_i) / sum(w_i) over present keys.

    Instances absent from `weights` are skipped. Instances absent from the
    score dict are skipped. If no keys overlap, falls through to combined_score.
    Negative weights raise (would invert the sign of the objective silently).
    """
    if any(w < 0 for w in weights.values()):
        raise ValueError("weighted: negative weights are not allowed")
    weights = dict(weights)  # defensive copy

    def _weighted(per_instance: Optional[Mapping[str, float]],
                  metrics: Mapping[str, float]) -> float:
        if not per_instance:
            return _fallback(metrics)
        num = 0.0
        den = 0.0
        for k, w in weights.items():
            v = per_instance.get(k)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(f):
                continue
            num += w * f
            den += w
        if den == 0.0:
            return _fallback(metrics)
        return num / den
    return _weighted


def pass_rate(threshold: float) -> Interpretation:
    """Fraction of finite per-instance scores >= threshold. Range [0, 1].

    `threshold` must be finite — NaN/inf would silently make every score
    fail or pass and is almost certainly a caller bug.
    """
    if not math.isfinite(float(threshold)):
        raise ValueError(f"pass_rate threshold must be finite, got {threshold!r}")

    def _pass_rate(per_instance: Optional[Mapping[str, float]],
                   metrics: Mapping[str, float]) -> float:
        vals = _finite_values(per_instance)
        if not vals:
            return _fallback(metrics)
        passing = sum(1 for v in vals if v >= threshold)
        return passing / len(vals)
    return _pass_rate


# ---------------------------------------------------------------------------
# YAML / CLI bridge — parse a string spec into an Interpretation
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, Callable[..., Interpretation]] = {
    "combined": combined,
    "mean": mean,
    "worst": worst,
    "best": best,
    "quantile": quantile,
    "cvar": cvar,
    "pass_rate": pass_rate,
}
# weighted() and combined(fallback=...) take non-scalar args (mapping / nested
# Interpretation) that don't round-trip through the simple "name(args)" string
# form. Users wanting them must construct via the Python API.


def from_spec(spec: str) -> Interpretation:
    """Parse a string like 'mean', 'cvar(0.1)', 'pass_rate(0.5)' into an
    Interpretation. Used by the YAML/CLI path where a Python callable can't
    be stored directly.

    Grammar: NAME | NAME() | NAME(ARG[, ARG]...) where ARGs are floats.
    Whitespace inside the parens is tolerated. Anything else raises ValueError.

    Supported names: combined, mean, worst, best, quantile, cvar, pass_rate.
    `weighted` and `combined(fallback=...)` are Python-API-only.
    """
    s = spec.strip()
    if "(" not in s:
        name, args_text = s, ""
    else:
        if not s.endswith(")"):
            raise ValueError(f"interpret spec missing closing paren: {spec!r}")
        name, _, rest = s.partition("(")
        args_text = rest[:-1]  # strip trailing ')'
    name = name.strip()
    factory = _REGISTRY.get(name)
    if factory is None:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"unknown interpret name {name!r}; supported: {valid}"
        )
    args_text = args_text.strip()
    if not args_text:
        return factory()
    try:
        args = [float(p.strip()) for p in args_text.split(",") if p.strip()]
    except ValueError as e:
        raise ValueError(
            f"interpret spec {spec!r}: arguments must be numeric ({e})"
        ) from e
    return factory(*args)


__all__ = [
    "Interpretation",
    "combined",
    "mean",
    "worst",
    "best",
    "quantile",
    "cvar",
    "weighted",
    "pass_rate",
    "from_spec",
]
