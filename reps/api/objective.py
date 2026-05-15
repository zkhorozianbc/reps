"""`reps.Objective` and `reps.LLMJudge` — the objective layer.

An `Objective` compiles a `(entrypoint, train_set, metric)` triple into the
existing evaluator contract: `Objective.evaluate(code: str) -> EvaluationResult`.
`Optimizer.optimize(objective=...)` registers `objective.evaluate` through the
same dispatch shim it uses for raw `evaluate=` callables.

See docs/objective_api_spec.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from reps.api.example import Example, Prediction, as_prediction
from reps.evaluation_result import EvaluationResult

MetricCallable = Callable[..., Any]

_DIRECTIONS = ("maximize", "minimize")
_NOT_IN_V1 = {"semantic_f1"}


# --- aggregation helpers ----------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rmse_aggregate(squared_errors: Sequence[float]) -> float:
    return math.sqrt(_mean(squared_errors))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _format_kwargs(kwargs: Mapping[str, Any]) -> str:
    """Render an inputs mapping as a call-arg string: `x=-4, hint='h'`."""
    return ", ".join(f"{key}={value!r}" for key, value in kwargs.items())


# --- built-in per-example metrics -------------------------------------------


def _accuracy(example: Example, pred: Prediction, trace: Any = None) -> float:
    return 1.0 if pred.get("answer") == example.get("answer") else 0.0


def _exact_match(example: Example, pred: Prediction, trace: Any = None) -> float:
    # Case-sensitive exact match for string answers.
    return 1.0 if str(pred.get("answer")) == str(example.get("answer")) else 0.0


def _abs_error(example: Example, pred: Prediction, trace: Any = None) -> float:
    return abs(float(pred.get("answer")) - float(example.get("answer")))


def _squared_error(example: Example, pred: Prediction, trace: Any = None) -> float:
    diff = float(pred.get("answer")) - float(example.get("answer"))
    return diff * diff


@dataclass(frozen=True)
class _BuiltinMetric:
    per_example: MetricCallable
    aggregate: Callable[[Sequence[float]], float]
    direction: str


_BUILTIN_METRICS: dict[str, _BuiltinMetric] = {
    "accuracy": _BuiltinMetric(_accuracy, _mean, "maximize"),
    "exact_match": _BuiltinMetric(_exact_match, _mean, "maximize"),
    "mae": _BuiltinMetric(_abs_error, _mean, "minimize"),
    "mse": _BuiltinMetric(_squared_error, _mean, "minimize"),
    "rmse": _BuiltinMetric(_squared_error, _rmse_aggregate, "minimize"),
}


def _resolve_metric(
    metric: "str | MetricCallable", direction: str
) -> "tuple[str, MetricCallable, Callable[[Sequence[float]], float]]":
    """Resolve `metric` into `(metric_name, per_example_fn, aggregate_fn)`.

    Built-in metric names are validated against `direction`; custom callables
    inherit `direction` from the calling classmethod with no validation.
    """
    if isinstance(metric, str):
        if metric in _NOT_IN_V1:
            raise ValueError(
                f"reps.Objective: metric {metric!r} is named in the spec but "
                f"not implemented in v1 of the objective layer. Pass a custom "
                f"metric callable `metric(example, pred, trace=None) -> float`."
            )
        builtin = _BUILTIN_METRICS.get(metric)
        if builtin is None:
            raise ValueError(
                f"reps.Objective: unknown metric {metric!r}. Built-in metrics: "
                f"{sorted(_BUILTIN_METRICS)}. Or pass a callable "
                f"`metric(example, pred, trace=None) -> float`."
            )
        if builtin.direction != direction:
            raise ValueError(
                f"reps.Objective: built-in metric {metric!r} is a "
                f"{builtin.direction} metric — use "
                f"`reps.Objective.{builtin.direction}(...)`."
            )
        return metric, builtin.per_example, builtin.aggregate
    if callable(metric):
        return getattr(metric, "__name__", "metric"), metric, _mean
    raise TypeError(
        f"reps.Objective: `metric` must be a built-in metric name (str) or a "
        f"callable, got {type(metric).__name__}"
    )


class Objective:
    """Deterministic objective layer for `reps.Optimizer.optimize(objective=...)`.

    Build one with `Objective.minimize(...)` or `Objective.maximize(...)`:

        objective = reps.Objective.minimize(
            entrypoint="predict",
            train_set=[reps.Example(x=-4, answer=30).with_inputs("x")],
            metric="mae",
        )

    `failure_score` is the per-example metric value (in the metric's natural
    space) assigned when an example's entrypoint call, prediction wrap, or
    metric call raises — and to every example when the candidate program
    fails to load. The `0.0` default suits maximize objectives; minimize
    objectives should pass a large positive value so crashing candidates are
    penalized rather than rewarded.
    """

    def __init__(
        self,
        *,
        entrypoint: str,
        train_set: "Sequence[Example | Mapping[str, Any]]",
        direction: str,
        metric_name: str,
        per_example_fn: "MetricCallable | None",
        aggregate_fn: Callable[[Sequence[float]], float],
        failure_score: float = 0.0,
    ) -> None:
        if direction not in _DIRECTIONS:
            raise ValueError(
                f"reps.Objective: direction must be one of {_DIRECTIONS}, got "
                f"{direction!r}"
            )
        if not entrypoint or not isinstance(entrypoint, str):
            raise ValueError(
                f"reps.Objective: `entrypoint` must be a non-empty function "
                f"name, got {entrypoint!r}"
            )
        if not train_set:
            raise ValueError("reps.Objective: `train_set` must be non-empty.")

        coerced: list[Example] = []
        for i, row in enumerate(train_set):
            ex = row if isinstance(row, Example) else Example(row)
            if not ex.input_keys:
                raise ValueError(
                    f"reps.Objective: train_set[{i}] has no input keys. Call "
                    f"`.with_inputs(...)` on each reps.Example — REPS does not "
                    f"infer input fields."
                )
            coerced.append(ex)

        ids = [str(ex["id"]) for ex in coerced if "id" in ex]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "reps.Objective: train_set has duplicate `id` fields; "
                "per-instance score keys must be unique."
            )

        self.entrypoint = entrypoint
        self.direction = direction
        self.failure_score = float(failure_score)
        self.train_set: list[Example] = coerced
        self.metric_name = metric_name
        self._per_example_fn = per_example_fn
        self._aggregate_fn = aggregate_fn

    @classmethod
    def maximize(
        cls,
        *,
        entrypoint: str,
        train_set: "Sequence[Example | Mapping[str, Any]]",
        metric: "str | MetricCallable",
        failure_score: float = 0.0,
    ) -> "Objective":
        name, per_ex, agg = _resolve_metric(metric, "maximize")
        return cls(
            entrypoint=entrypoint,
            train_set=train_set,
            direction="maximize",
            metric_name=name,
            per_example_fn=per_ex,
            aggregate_fn=agg,
            failure_score=failure_score,
        )

    @classmethod
    def minimize(
        cls,
        *,
        entrypoint: str,
        train_set: "Sequence[Example | Mapping[str, Any]]",
        metric: "str | MetricCallable",
        failure_score: float = 0.0,
    ) -> "Objective":
        name, per_ex, agg = _resolve_metric(metric, "minimize")
        return cls(
            entrypoint=entrypoint,
            train_set=train_set,
            direction="minimize",
            metric_name=name,
            per_example_fn=per_ex,
            aggregate_fn=agg,
            failure_score=failure_score,
        )

    # --- evaluation ---------------------------------------------------------

    def evaluate(self, code: str) -> EvaluationResult:
        """Run candidate `code` against the train set; return an
        `EvaluationResult` in higher-is-better `combined_score` space.

        Note: `code` is exec'd in-process (same trust model as a hand-written
        `evaluate=` callable). Sandbox untrusted candidates upstream.
        """
        entry = self._load_entrypoint(code)
        if entry is None:
            raw = [
                (self._example_key(ex, i), self.failure_score)
                for i, ex in enumerate(self.train_set)
            ]
            return self._build_result(
                raw=raw,
                failures=[
                    f"entrypoint {self.entrypoint!r} could not be loaded from "
                    f"the candidate program"
                ],
            )

        raw: list[tuple[str, float]] = []
        failures: list[str] = []
        detail: list[str] = []
        for i, ex in enumerate(self.train_set):
            key = self._example_key(ex, i)
            inputs = ex.inputs().to_dict()
            call = f"{self.entrypoint}({_format_kwargs(inputs)})"
            expected = ex.get("answer")
            try:
                prediction = as_prediction(entry(**inputs))
                value = float(self._per_example_fn(ex, prediction, None))
                detail.append(
                    f"{key}: {call} -> {prediction.get('answer')!r} | "
                    f"expected {expected!r} | {self.metric_name} {value:g}"
                )
            except Exception as exc:  # candidate code is untrusted
                value = self.failure_score
                failures.append(f"{key}: {type(exc).__name__}: {exc}")
                detail.append(
                    f"{key}: {call} raised {type(exc).__name__}: {exc} | "
                    f"expected {expected!r}"
                )
            raw.append((key, value))
        return self._build_result(raw=raw, failures=failures, detail=detail)

    def _load_entrypoint(self, code: str):
        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)  # candidate code, same trust model as evaluate=
        except Exception:
            return None
        entry = namespace.get(self.entrypoint)
        return entry if callable(entry) else None

    def _example_key(self, example: Example, index: int) -> str:
        if "id" in example:
            return str(example["id"])
        return f"train/{index}"

    def _build_result(
        self,
        *,
        raw: list,
        failures: list,
        detail: "list | None" = None,
        rationales: "list | None" = None,
    ) -> EvaluationResult:
        values = [v for _, v in raw]
        total = len(values)
        validity = (total - len(failures)) / total if total else 0.0
        aggregate = self._aggregate_fn(values)

        if self.direction == "maximize":
            combined = aggregate
            per_instance = {k: v for k, v in raw}
        else:
            # `0.0 - x` rather than `-x` so a perfect run yields +0.0, not -0.0.
            combined = 0.0 - aggregate
            per_instance = {k: 0.0 - v for k, v in raw}

        metrics = {
            "combined_score": combined,
            self.metric_name: aggregate,
            "validity": validity,
        }
        return EvaluationResult(
            metrics=metrics,
            per_instance_scores=per_instance,
            feedback=self._build_feedback(failures, detail, rationales),
        )

    def _build_feedback(
        self,
        failures: list,
        detail: "list | None" = None,
        rationales: "list | None" = None,
    ) -> "str | None":
        # The per-example detail is the load-bearing signal for the mutation
        # LLM: it shows input -> predicted vs expected -> metric per row, so
        # the model can see the actual pattern instead of guessing from an
        # aggregate score.
        parts: list[str] = []
        if detail:
            parts.append("per-example results:\n" + "\n".join(detail))
        if rationales:
            parts.append("judge rationales:\n" + "\n".join(rationales))
        if failures:
            parts.append("failures: " + "; ".join(failures))
        return "\n".join(parts) if parts else None


class LLMJudge(Objective):
    """An `Objective` that scores candidates with an LLM judge.

    For long-form or subjective outputs where deterministic metrics aren't
    enough. Prefer a deterministic `reps.Objective` when you can — an LLM
    judge is a useful but imperfect metric.

        judge = reps.LLMJudge(
            entrypoint="answer",
            train_set=[reps.Example(question="...", answer="...").with_inputs("question")],
            rubric="Score factual correctness, completeness, and concision.",
            model="openai/gpt-5.1-mini",
            scale=(0.0, 1.0),
        )

    `model` may be a model-name string (a `reps.Model` is built lazily on
    first use with `temperature=0`) or any `(prompt: str) -> str` callable
    (e.g. a `reps.Model` instance, or a fake for tests). Judge calls are
    cached by `(code_hash, example_key, rubric_hash, model_key)`.
    """

    def __init__(
        self,
        *,
        entrypoint: str,
        train_set: "Sequence[Example | Mapping[str, Any]]",
        rubric: str,
        model: "str | Callable[[str], str]",
        scale: "tuple[float, float]" = (0.0, 1.0),
        failure_score: float = 0.0,
    ) -> None:
        if not rubric or not isinstance(rubric, str):
            raise ValueError("reps.LLMJudge: `rubric` must be a non-empty string.")
        low, high = scale
        if not low < high:
            raise ValueError(
                f"reps.LLMJudge: `scale` must be (low, high) with low < high, "
                f"got {scale!r}"
            )
        self.rubric = rubric
        self.scale = (float(low), float(high))
        self._model_spec = model
        self._judge_callable: "Callable[[str], str] | None" = None
        self._cache: dict[tuple, tuple[float, str]] = {}
        super().__init__(
            entrypoint=entrypoint,
            train_set=train_set,
            direction="maximize",
            metric_name="judge_score",
            per_example_fn=None,  # LLMJudge fully overrides evaluate()
            aggregate_fn=_mean,
            failure_score=failure_score,
        )

    def evaluate(self, code: str) -> EvaluationResult:
        """Score `code` by running its entrypoint and grading each output
        with the judge model. Fully overrides `Objective.evaluate` so all
        per-call state stays local — the objective instance is shared across
        parallel evaluations."""
        entry = self._load_entrypoint(code)
        if entry is None:
            raw = [
                (self._example_key(ex, i), self.failure_score)
                for i, ex in enumerate(self.train_set)
            ]
            return self._build_result(
                raw=raw,
                failures=[
                    f"entrypoint {self.entrypoint!r} could not be loaded from "
                    f"the candidate program"
                ],
                rationales=[],
            )

        code_hash = _hash_text(code)
        rubric_hash = _hash_text(self.rubric)
        model_key = self._model_key()
        judge = self._get_judge()

        raw: list[tuple[str, float]] = []
        failures: list[str] = []
        rationales: list[str] = []
        for i, ex in enumerate(self.train_set):
            key = self._example_key(ex, i)
            try:
                prediction = as_prediction(entry(**ex.inputs().to_dict()))
            except Exception as exc:
                raw.append((key, self.failure_score))
                failures.append(
                    f"{key}: entrypoint raised {type(exc).__name__}: {exc}"
                )
                continue

            cache_key = (code_hash, key, rubric_hash, model_key)
            cached = self._cache.get(cache_key)
            if cached is not None:
                score, rationale = cached
            else:
                try:
                    response = judge(self._build_judge_prompt(ex, prediction))
                    score, rationale = self._parse_judge_score(response)
                except Exception as exc:
                    raw.append((key, self.failure_score))
                    failures.append(
                        f"{key}: judge call failed: {type(exc).__name__}: {exc}"
                    )
                    continue
                self._cache[cache_key] = (score, rationale)

            raw.append((key, score))
            rationales.append(f"{key}: {rationale}")
        return self._build_result(raw=raw, failures=failures, rationales=rationales)

    # --- judge plumbing -----------------------------------------------------

    def _model_key(self) -> str:
        spec = self._model_spec
        return spec if isinstance(spec, str) else repr(spec)

    def _get_judge(self) -> "Callable[[str], str]":
        """Resolve the judge into a `(prompt: str) -> str` callable, lazily.

        A callable (including a `reps.Model`) is used directly; a model-name
        string builds a `reps.Model(..., temperature=0)`. Lazy so
        `LLMJudge(model="openai/...")` can be constructed without the
        provider API key (e.g. in tests).
        """
        if self._judge_callable is not None:
            return self._judge_callable
        spec = self._model_spec
        if callable(spec):
            self._judge_callable = spec
        elif isinstance(spec, str):
            from reps.api.model import Model

            self._judge_callable = Model(spec, temperature=0)
        else:
            raise TypeError(
                f"reps.LLMJudge: `model` must be a model-name string or a "
                f"`(prompt: str) -> str` callable, got {type(spec).__name__}"
            )
        return self._judge_callable

    def _build_judge_prompt(self, example: Example, pred: Prediction) -> str:
        inputs = example.inputs().to_dict()
        labels = example.labels().to_dict()
        low, high = self.scale
        return (
            "You are grading the output of a candidate program.\n\n"
            f"Rubric:\n{self.rubric}\n\n"
            f"Program inputs:\n{json.dumps(inputs, default=str, indent=2)}\n\n"
            "Expected / reference fields:\n"
            f"{json.dumps(labels, default=str, indent=2)}\n\n"
            "Candidate output:\n"
            f"{json.dumps(pred.to_dict(), default=str, indent=2)}\n\n"
            f"Score the candidate output from {low} to {high} according to the "
            "rubric. Respond with a single JSON object: "
            '{"score": <number>, "rationale": "<one or two sentences>"}'
        )

    def _parse_judge_score(self, response: str) -> "tuple[float, str]":
        low, high = self.scale
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                score = float(obj["score"])
                rationale = str(obj.get("rationale", "")).strip()
                return _clamp(score, low, high), rationale
            except (ValueError, KeyError, TypeError):
                pass
        number = re.search(r"-?\d+(?:\.\d+)?", response)
        if number:
            return _clamp(float(number.group(0)), low, high), response.strip()[:200]
        raise ValueError(
            f"reps.LLMJudge: could not parse a numeric score from judge "
            f"response: {response[:200]!r}"
        )
