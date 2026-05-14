"""Unit tests for `reps.Objective` and `reps.LLMJudge`."""

from __future__ import annotations

import math

import pytest

import reps
from reps.api.example import Example, Prediction
from reps.api.objective import LLMJudge, Objective, _BUILTIN_METRICS, _resolve_metric


# --- built-in metric registry ----------------------------------------------


def test_builtin_metric_names_and_directions():
    assert _BUILTIN_METRICS["accuracy"].direction == "maximize"
    assert _BUILTIN_METRICS["exact_match"].direction == "maximize"
    assert _BUILTIN_METRICS["mae"].direction == "minimize"
    assert _BUILTIN_METRICS["mse"].direction == "minimize"
    assert _BUILTIN_METRICS["rmse"].direction == "minimize"


def test_accuracy_per_example():
    fn = _BUILTIN_METRICS["accuracy"].per_example
    assert fn(Example(answer="pos"), Prediction(answer="pos")) == 1.0
    assert fn(Example(answer="pos"), Prediction(answer="neg")) == 0.0


def test_mae_per_example_and_aggregate():
    m = _BUILTIN_METRICS["mae"]
    assert m.per_example(Example(answer=2.0), Prediction(answer=5.0)) == 3.0
    assert m.aggregate([1.0, 3.0]) == 2.0


def test_mse_per_example_and_aggregate():
    m = _BUILTIN_METRICS["mse"]
    assert m.per_example(Example(answer=2.0), Prediction(answer=5.0)) == 9.0
    assert m.aggregate([1.0, 9.0]) == 5.0


def test_rmse_aggregate_is_sqrt_of_mean_squared_error():
    m = _BUILTIN_METRICS["rmse"]
    # per_example returns squared error; aggregate takes sqrt(mean).
    assert m.per_example(Example(answer=0.0), Prediction(answer=3.0)) == 9.0
    assert m.aggregate([9.0, 16.0]) == math.sqrt(12.5)


# --- _resolve_metric --------------------------------------------------------


def test_resolve_builtin_metric_matching_direction():
    name, per_ex, agg = _resolve_metric("mae", "minimize")
    assert name == "mae"
    assert per_ex is _BUILTIN_METRICS["mae"].per_example
    assert agg is _BUILTIN_METRICS["mae"].aggregate


def test_resolve_builtin_metric_direction_mismatch_raises():
    with pytest.raises(ValueError, match="minimize metric"):
        _resolve_metric("mae", "maximize")
    with pytest.raises(ValueError, match="maximize metric"):
        _resolve_metric("accuracy", "minimize")


def test_resolve_unknown_metric_name_raises():
    with pytest.raises(ValueError, match="unknown metric"):
        _resolve_metric("f1_score", "maximize")


def test_resolve_semantic_f1_not_in_v1():
    with pytest.raises(ValueError, match="not implemented in v1"):
        _resolve_metric("semantic_f1", "maximize")


def test_resolve_custom_callable():
    def close_enough(example, pred, trace=None):
        return abs(float(pred.answer) - float(example.answer)) <= 0.1

    name, per_ex, agg = _resolve_metric(close_enough, "maximize")
    assert name == "close_enough"
    assert per_ex is close_enough


def test_resolve_rejects_non_str_non_callable():
    with pytest.raises(TypeError, match="must be a built-in metric name"):
        _resolve_metric(123, "maximize")  # type: ignore[arg-type]


# --- Objective construction -------------------------------------------------


def _train_set():
    return [
        Example(x=-4, answer=30).with_inputs("x"),
        Example(x=0, answer=2).with_inputs("x"),
    ]


def test_minimize_classmethod_builds_objective():
    obj = Objective.minimize(entrypoint="predict", train_set=_train_set(), metric="mae")
    assert obj.direction == "minimize"
    assert obj.entrypoint == "predict"
    assert obj.metric_name == "mae"
    assert len(obj.train_set) == 2
    assert obj.failure_score == 0.0


def test_maximize_classmethod_builds_objective():
    obj = Objective.maximize(
        entrypoint="classify",
        train_set=[Example(text="hi", answer="pos").with_inputs("text")],
        metric="accuracy",
    )
    assert obj.direction == "maximize"
    assert obj.metric_name == "accuracy"


def test_objective_coerces_mapping_rows_to_examples():
    # A plain mapping row is accepted, but still needs explicit input keys.
    obj = Objective.minimize(
        entrypoint="predict",
        train_set=[Example({"x": 1, "answer": 2}).with_inputs("x")],
        metric="mae",
    )
    assert isinstance(obj.train_set[0], Example)


def test_objective_rejects_row_without_input_keys():
    with pytest.raises(ValueError, match="no input keys"):
        Objective.minimize(
            entrypoint="predict",
            train_set=[{"x": 1, "answer": 2}],  # bare mapping, no .with_inputs
            metric="mae",
        )


def test_objective_rejects_empty_train_set():
    with pytest.raises(ValueError, match="non-empty"):
        Objective.minimize(entrypoint="predict", train_set=[], metric="mae")


def test_objective_rejects_blank_entrypoint():
    with pytest.raises(ValueError, match="entrypoint"):
        Objective.minimize(entrypoint="", train_set=_train_set(), metric="mae")


def test_objective_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate `id`"):
        Objective.minimize(
            entrypoint="predict",
            train_set=[
                Example(id="dup", x=1, answer=2).with_inputs("x"),
                Example(id="dup", x=2, answer=3).with_inputs("x"),
            ],
            metric="mae",
        )


def test_objective_custom_metric_callable():
    def close_enough(example, pred, trace=None):
        return abs(float(pred.answer) - float(example.answer)) <= 0.1

    obj = Objective.maximize(
        entrypoint="predict", train_set=_train_set(), metric=close_enough
    )
    assert obj.metric_name == "close_enough"
    assert obj.direction == "maximize"


def test_objective_direction_mismatch_raises_via_classmethod():
    with pytest.raises(ValueError, match="minimize metric"):
        Objective.maximize(entrypoint="predict", train_set=_train_set(), metric="mae")


# --- Objective.evaluate -----------------------------------------------------

_QUADRATIC = "def predict(x):\n    return x * x - 3 * x + 2\n"
_IDENTITY = "def predict(x):\n    return x\n"


def test_evaluate_minimize_perfect_program():
    obj = Objective.minimize(
        entrypoint="predict",
        train_set=[
            Example(x=-4, answer=30).with_inputs("x"),
            Example(x=0, answer=2).with_inputs("x"),
        ],
        metric="mae",
    )
    result = obj.evaluate(_QUADRATIC)
    assert isinstance(result, reps.EvaluationResult)
    assert result.metrics["mae"] == 0.0
    assert result.metrics["combined_score"] == 0.0
    assert result.metrics["validity"] == 1.0
    assert result.per_instance_scores == {"train/0": 0.0, "train/1": 0.0}


def test_evaluate_minimize_imperfect_program_negates_loss():
    obj = Objective.minimize(
        entrypoint="predict",
        train_set=[
            Example(x=0, answer=2).with_inputs("x"),
            Example(x=3, answer=2).with_inputs("x"),
        ],
        metric="mae",
    )
    # predict(x)=x -> errors |0-2|=2, |3-2|=1 -> mae=1.5
    result = obj.evaluate(_IDENTITY)
    assert result.metrics["mae"] == 1.5
    assert result.metrics["combined_score"] == -1.5
    # per-instance is higher-is-better: -raw error
    assert result.per_instance_scores == {"train/0": -2.0, "train/1": -1.0}
    assert "raw mae" in result.feedback


def test_evaluate_maximize_accuracy():
    obj = Objective.maximize(
        entrypoint="classify",
        train_set=[
            Example(text="a", answer="pos").with_inputs("text"),
            Example(text="b", answer="neg").with_inputs("text"),
        ],
        metric="accuracy",
    )
    code = "def classify(text):\n    return 'pos'\n"  # right on 1 of 2
    result = obj.evaluate(code)
    assert result.metrics["accuracy"] == 0.5
    assert result.metrics["combined_score"] == 0.5
    assert result.per_instance_scores == {"train/0": 1.0, "train/1": 0.0}


def test_evaluate_uses_example_id_for_per_instance_keys():
    obj = Objective.minimize(
        entrypoint="predict",
        train_set=[
            Example(id="easy", x=0, answer=0).with_inputs("x"),
            Example(id="hard", x=5, answer=0).with_inputs("x"),
        ],
        metric="mae",
    )
    result = obj.evaluate(_IDENTITY)
    assert set(result.per_instance_scores) == {"easy", "hard"}


def test_evaluate_program_fails_to_load():
    obj = Objective.maximize(
        entrypoint="predict",
        train_set=[Example(x=1, answer=1).with_inputs("x")],
        metric="accuracy",
    )
    result = obj.evaluate("this is not valid python !!!")
    assert result.metrics["combined_score"] == 0.0  # failure_score
    assert result.metrics["validity"] == 0.0
    assert "could not be loaded" in result.feedback


def test_evaluate_entrypoint_missing():
    obj = Objective.maximize(
        entrypoint="predict",
        train_set=[Example(x=1, answer=1).with_inputs("x")],
        metric="accuracy",
    )
    result = obj.evaluate("def something_else():\n    return 1\n")
    assert result.metrics["validity"] == 0.0
    assert "could not be loaded" in result.feedback


def test_evaluate_per_example_failure_uses_failure_score():
    obj = Objective.maximize(
        entrypoint="predict",
        train_set=[
            Example(x=1, answer=1).with_inputs("x"),
            Example(x=0, answer=1).with_inputs("x"),
        ],
        metric="accuracy",
        failure_score=0.0,
    )
    # predict raises on x==0 (ZeroDivisionError); scores 1.0 on x==1.
    code = "def predict(x):\n    return 1 // x\n"
    result = obj.evaluate(code)
    assert result.metrics["accuracy"] == 0.5  # (1.0 + 0.0) / 2
    assert result.metrics["validity"] == 0.5  # 1 of 2 examples ran
    assert "failures:" in result.feedback


def test_evaluate_custom_metric_callable():
    def close_enough(example, pred, trace=None):
        return abs(float(pred.answer) - float(example.answer)) <= 0.5

    obj = Objective.maximize(
        entrypoint="predict",
        train_set=[
            Example(x=2, answer=2.0).with_inputs("x"),
            Example(x=9, answer=2.0).with_inputs("x"),
        ],
        metric=close_enough,
    )
    result = obj.evaluate(_IDENTITY)  # predict(x)=x -> close on x=2, far on x=9
    assert result.metrics["close_enough"] == 0.5
    assert result.metrics["combined_score"] == 0.5


# --- LLMJudge ---------------------------------------------------------------


def _judge_train_set():
    return [
        Example(question="What is REPS?", answer="A program search harness.")
        .with_inputs("question")
    ]


def test_llm_judge_is_objective_subclass():
    judge = LLMJudge(
        entrypoint="answer",
        train_set=_judge_train_set(),
        rubric="score correctness",
        model=lambda prompt: '{"score": 1.0, "rationale": "ok"}',
    )
    assert isinstance(judge, Objective)
    assert judge.direction == "maximize"
    assert judge.metric_name == "judge_score"


def test_llm_judge_rejects_blank_rubric():
    with pytest.raises(ValueError, match="rubric"):
        LLMJudge(
            entrypoint="answer",
            train_set=_judge_train_set(),
            rubric="",
            model=lambda p: "{}",
        )


def test_llm_judge_rejects_bad_scale():
    with pytest.raises(ValueError, match="scale"):
        LLMJudge(
            entrypoint="answer",
            train_set=_judge_train_set(),
            rubric="r",
            model=lambda p: "{}",
            scale=(1.0, 0.0),
        )


def test_llm_judge_scores_with_fake_callable():
    calls = []

    def fake_judge(prompt: str) -> str:
        calls.append(prompt)
        return '{"score": 0.8, "rationale": "good and concise"}'

    judge = LLMJudge(
        entrypoint="answer",
        train_set=_judge_train_set(),
        rubric="Score factual correctness and concision.",
        model=fake_judge,
    )
    result = judge.evaluate("def answer(question):\n    return 'REPS'\n")
    assert result.metrics["combined_score"] == 0.8
    assert result.metrics["judge_score"] == 0.8
    assert result.metrics["validity"] == 1.0
    assert result.per_instance_scores == {"train/0": 0.8}
    assert "good and concise" in result.feedback
    assert len(calls) == 1
    # Prompt construction includes the rubric and the candidate output.
    assert "Score factual correctness and concision." in calls[0]
    assert "REPS" in calls[0]


def test_llm_judge_caches_repeated_evaluations():
    calls = []

    def fake_judge(prompt: str) -> str:
        calls.append(prompt)
        return '{"score": 1.0, "rationale": "ok"}'

    judge = LLMJudge(
        entrypoint="answer",
        train_set=_judge_train_set(),
        rubric="r",
        model=fake_judge,
    )
    code = "def answer(question):\n    return 'x'\n"
    judge.evaluate(code)
    judge.evaluate(code)  # identical code+example+rubric+model -> cache hit
    assert len(calls) == 1


def test_llm_judge_parses_bare_number_response():
    judge = LLMJudge(
        entrypoint="answer",
        train_set=_judge_train_set(),
        rubric="r",
        model=lambda prompt: "I'd give this a 0.6 overall.",
    )
    result = judge.evaluate("def answer(question):\n    return 'x'\n")
    assert result.metrics["judge_score"] == 0.6


def test_llm_judge_clamps_to_scale():
    judge = LLMJudge(
        entrypoint="answer",
        train_set=_judge_train_set(),
        rubric="r",
        model=lambda prompt: '{"score": 9.0, "rationale": "over"}',
        scale=(0.0, 1.0),
    )
    result = judge.evaluate("def answer(question):\n    return 'x'\n")
    assert result.metrics["judge_score"] == 1.0


def test_llm_judge_entrypoint_failure_uses_failure_score():
    judge = LLMJudge(
        entrypoint="answer",
        train_set=_judge_train_set(),
        rubric="r",
        model=lambda prompt: '{"score": 1.0, "rationale": "ok"}',
    )
    result = judge.evaluate("def answer(question):\n    raise RuntimeError('boom')\n")
    assert result.metrics["combined_score"] == 0.0  # failure_score
    assert result.metrics["validity"] == 0.0
    assert "failures:" in result.feedback


def test_llm_judge_string_model_builds_reps_model_lazily(monkeypatch):
    # Constructing LLMJudge with a model string must NOT require an API key —
    # the reps.Model is built lazily on first evaluate().
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judge = LLMJudge(
        entrypoint="answer",
        train_set=_judge_train_set(),
        rubric="r",
        model="openai/gpt-5.1-mini",
    )
    # Lazy: no Model built yet, so no key needed at construction time.
    assert judge._judge_callable is None


# --- re-exports -------------------------------------------------------------


def test_top_level_reexports():
    from reps.api.example import Example as _Example
    from reps.api.example import Prediction as _Prediction
    from reps.api.objective import LLMJudge as _LLMJudge
    from reps.api.objective import Objective as _Objective

    assert reps.Example is _Example
    assert reps.Prediction is _Prediction
    assert reps.Objective is _Objective
    assert reps.LLMJudge is _LLMJudge


def test_api_package_reexports():
    import reps.api as api

    assert api.Example is reps.Example
    assert api.Prediction is reps.Prediction
    assert api.Objective is reps.Objective
    assert api.LLMJudge is reps.LLMJudge


def test_new_symbols_in_dunder_all():
    for name in ("Example", "Prediction", "Objective", "LLMJudge"):
        assert name in reps.__all__
