# REPS Objective API Spec

> **Status: implemented.** See [`reps/api/example.py`](../reps/api/example.py),
> [`reps/api/objective.py`](../reps/api/objective.py),
> [`tests/test_api_example.py`](../tests/test_api_example.py),
> [`tests/test_api_objective.py`](../tests/test_api_objective.py), and
> [`tests/test_api_objective_integration.py`](../tests/test_api_objective_integration.py).

## Goal

Make REPS easy to start without asking users to hand-write an
`evaluate(code)` function or manually `exec` generated code. The low-level
evaluator contract stays available for power users, but the primary API should
be:

```python
result = reps.Optimizer(model="anthropic/claude-sonnet-4-6").optimize(
    initial=seed_program,
    objective=reps.Objective.minimize(
        entrypoint="predict",
        train_set=[
            reps.Example(x=-4, answer=30).with_inputs("x"),
            reps.Example(x=0, answer=2).with_inputs("x"),
            reps.Example(x=3, answer=2).with_inputs("x"),
        ],
        metric="mae",
    ),
)
```

The mental model is:

```text
seed program + train_set + metric -> best evolved program
```

## Design Principles

1. **Simple first run**: users should not need to know about evaluator files,
   temp files, `exec`, or `EvaluationResult` to get a useful run.
2. **DSPy-inspired, not DSPy-shaped**: borrow `Example`, `train_set`, metric
   callables, and LLM-as-judge metrics, but do not require `Module`,
   `Signature`, `Predict`, or `.compile(...)`.
3. **Industry-standard scoring semantics**: every score that reaches the REPS
   engine is higher-is-better. Losses are stored as raw metrics, then converted
   to a higher-is-better `combined_score`.
4. **Examples are records**: no tuple-of-tuples in user-facing docs. Each data
   row has named fields and explicit input keys.
5. **Escape hatches remain**: `evaluate=` continues to accept the current
   `Callable[[str], float | dict | EvaluationResult]` shape.

## External Grounding

- DSPy uses `Example` objects as training/evaluation rows. An `Example` is a
  dict-like record; `.with_inputs(...)` marks which fields are inputs, while
  remaining fields are labels or metadata. See
  [DSPy data handling](https://dspy.ai/learn/evaluation/data/).
- DSPy metrics are functions over `example`, prediction, and optional `trace`,
  returning `bool`, `int`, or `float`. See
  [DSPy metrics](https://dspy.ai/learn/evaluation/metrics/).
- DSPy optimizers are commonly called with `trainset=...` and `metric=...`.
  See [DSPy optimizers](https://dspy.ai/learn/optimization/optimizers/).
- scikit-learn scoring supports string metric names and custom callable
  scorers, with higher return values treated as better. Loss metrics are
  commonly negated when exposed through scorer APIs. See
  [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).
- SciPy uses the term "objective function" for functions being optimized. See
  [SciPy optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html).

## Proposed Public Surface

### `reps.Example`

Small data primitive inspired by `dspy.Example`.

```python
class Example:
    def __init__(self, base: Mapping[str, Any] | None = None, **fields: Any) -> None: ...
    def with_inputs(self, *keys: str) -> "Example": ...
    def inputs(self) -> "Example": ...
    def labels(self) -> "Example": ...
    def to_dict(self) -> dict[str, Any]: ...
```

Behavior:

- Constructed from keyword fields or a mapping:

  ```python
  reps.Example(question="What is REPS?", answer="A program search harness.")
  reps.Example({"x": 3, "answer": 2})
  ```

- Supports dot access and dict-like access:

  ```python
  example.question
  example["question"]
  ```

- `.with_inputs("field", ...)` returns a copy with explicit input keys.
- `.inputs()` returns a record containing only input fields.
- `.labels()` returns a record containing non-input fields.
- Calling `.inputs()` or `.labels()` without input keys raises a clear
  `ValueError`.

Rationale: explicit `.with_inputs(...)` is slightly more verbose, but it is
standard in DSPy and prevents ambiguity when examples include fields such as
`context`, `hint`, `answer`, and `source`.

### `reps.Prediction`

Small dict-like output wrapper used by metrics.

```python
class Prediction(Example):
    pass
```

When an entrypoint returns:

- a scalar, wrap it as `Prediction(answer=value)`;
- a dict, wrap it as `Prediction(**value)`;
- an object with a `to_dict()` or `model_dump()` method, wrap that mapping;
- an existing `Prediction`, keep it.

Rationale: DSPy metrics compare `example.answer` to `pred.answer`. REPS should
make the same metric style natural even when candidate programs return plain
Python values.

### `reps.Objective`

The deterministic objective layer. It compiles to the current evaluator
contract internally.

```python
class Objective:
    @classmethod
    def maximize(
        cls,
        *,
        entrypoint: str,
        train_set: Sequence[Example | Mapping[str, Any]],
        metric: str | MetricCallable,
        failure_score: float = 0.0,
    ) -> "Objective": ...

    @classmethod
    def minimize(
        cls,
        *,
        entrypoint: str,
        train_set: Sequence[Example | Mapping[str, Any]],
        metric: str | MetricCallable,
        failure_score: float = 0.0,
    ) -> "Objective": ...

    def evaluate(self, code: str) -> EvaluationResult: ...
```

`MetricCallable` follows the DSPy-compatible shape:

```python
def metric(example: reps.Example, pred: reps.Prediction, trace=None) -> bool | int | float:
    ...
```

Accepted built-in metric names:

| Name | Direction | Meaning |
|---|---|---|
| `"accuracy"` | maximize | `pred.answer == example.answer` averaged over examples |
| `"exact_match"` | maximize | case-sensitive exact match for string answers |
| `"semantic_f1"` | maximize | optional later feature, not in v1 of this layer unless already implemented |
| `"mae"` | minimize | mean absolute error over numeric answers |
| `"mse"` | minimize | mean squared error over numeric answers |
| `"rmse"` | minimize | root mean squared error over numeric answers |

Direction handling:

- `Objective.maximize(...)`: aggregate metric scores directly into
  `combined_score`.
- `Objective.minimize(...)`: aggregate the raw loss into `metrics[metric_name]`
  and use `combined_score = -loss` internally, matching the higher-is-better
  scorer convention used by libraries like scikit-learn.
- `OptimizationResult.best_score` remains `combined_score`.
- `OptimizationResult.best_metrics` exposes the raw metric or loss so users do
  not have to read negated values.

Per-example reporting:

- `per_instance_scores` is keyed by stable example ids:

  ```python
  "train/0", "train/1", ...
  ```

- If an `Example` has an `id` field, prefer that:

  ```python
  reps.Example(id="hard_negative_7", text="...", answer="...")
  ```

- For minimize objectives, per-instance scores use the same higher-is-better
  conversion as `combined_score`, while raw losses are included in `feedback`.

### `reps.LLMJudge`

An objective for long-form or subjective outputs where deterministic metrics
are insufficient.

```python
judge = reps.LLMJudge(
    entrypoint="answer",
    train_set=[
        reps.Example(
            question="What is REPS?",
            answer="A recursive evolutionary program search harness.",
        ).with_inputs("question"),
    ],
    rubric="Score factual correctness, completeness, and concision.",
    model="openai/gpt-5.1-mini",
    scale=(0.0, 1.0),
)
```

Behavior:

- Calls the candidate entrypoint with `example.inputs()`.
- Wraps the result as `Prediction`.
- Sends the example inputs, expected labels, prediction, and rubric to the
  judge model.
- Returns a numeric score on `scale`.
- Stores judge rationale in `feedback`.
- Uses `temperature=0` by default for reproducibility.
- Caches judge calls by `(candidate_hash, example_id, rubric_hash, model)`.

`LLMJudge` is still an `Objective`, so users pass it the same way:

```python
result = reps.Optimizer(model="anthropic/claude-sonnet-4-6").optimize(
    initial=seed_program,
    objective=judge,
)
```

Guardrails:

- Docs should recommend deterministic metrics first when possible.
- LLM judges should be framed as useful but imperfect metrics, especially for
  subjective tasks.
- The judge model should be configurable independently from the mutation model.

## Optimizer API Change

Current:

```python
result = optimizer.optimize(initial=seed, evaluate=evaluate)
```

Proposed:

```python
result = optimizer.optimize(initial=seed, objective=objective)
```

Backwards-compatible signature:

```python
def optimize(
    self,
    initial: str,
    evaluate: Callable[..., Any] | None = None,
    *,
    objective: Objective | None = None,
    seed: int | None = None,
) -> OptimizationResult:
    ...
```

Rules:

- Exactly one of `evaluate` or `objective` must be supplied.
- Passing `evaluate` keeps current behavior.
- Passing `objective` registers `objective.evaluate` through the existing
  `evaluate_dispatch` shim.
- Passing `objective` defaults `trace_reflection` on (the objective always
  emits per-example feedback + per-instance scores the reflection path
  consumes — without it that signal is computed and dropped). An explicit
  `trace_reflection=True/False` on the `Optimizer` still wins.
- Error messages should steer new users toward `objective=` and power users
  toward `evaluate=`.

## User-Facing Examples

### Numeric Function Search

```python
import reps

seed = """
def predict(x):
    return x
"""

train_set = [
    reps.Example(x=-4, answer=30).with_inputs("x"),
    reps.Example(x=0, answer=2).with_inputs("x"),
    reps.Example(x=3, answer=2).with_inputs("x"),
    reps.Example(x=5, answer=12).with_inputs("x"),
]

result = reps.Optimizer(
    model="anthropic/claude-sonnet-4-6",
    max_iterations=20,
).optimize(
    initial=seed,
    objective=reps.Objective.minimize(
        entrypoint="predict",
        train_set=train_set,
        metric="mae",
    ),
)
```

### Classification

```python
train_set = [
    reps.Example(text="I loved it", answer="positive").with_inputs("text"),
    reps.Example(text="Never again", answer="negative").with_inputs("text"),
]

objective = reps.Objective.maximize(
    entrypoint="classify",
    train_set=train_set,
    metric="accuracy",
)
```

### LLM-as-Judge

```python
train_set = [
    reps.Example(
        question="What does REPS optimize?",
        answer="Python programs.",
    ).with_inputs("question"),
]

objective = reps.LLMJudge(
    entrypoint="answer",
    train_set=train_set,
    rubric="Score whether the answer is factually correct and concise.",
    model="openai/gpt-5.1-mini",
)
```

### Custom Metric

```python
def close_enough(example, pred, trace=None):
    return abs(float(pred.answer) - float(example.answer)) <= 0.1

objective = reps.Objective.maximize(
    entrypoint="predict",
    train_set=train_set,
    metric=close_enough,
)
```

## Implementation Sketch

1. Add `reps/api/example.py` with `Example` and `Prediction`.
2. Add `reps/api/objective.py` with `Objective`, built-in metric registry, and
   `LLMJudge`.
3. Re-export `Example`, `Prediction`, `Objective`, and `LLMJudge` from
   `reps/__init__.py` and `reps/api/__init__.py`.
4. Update `Optimizer.optimize(...)` to accept `objective=`.
5. Add tests:
   - `Example.with_inputs`, `.inputs`, `.labels`, dict/dot access.
   - scalar/dict/object entrypoint output wrapping into `Prediction`.
   - maximize objective with `accuracy`.
   - minimize objective with `mae`, including raw metric plus negated
     `combined_score`.
   - custom metric callable with optional `trace`.
   - exactly-one-of `evaluate` / `objective` validation.
   - `LLMJudge` prompt construction using a fake model client.
6. Update examples and README so the first-run path uses `objective=`, not
   handwritten `evaluate(code)`.

## Compatibility

- Existing users of `evaluate=` are unaffected.
- Existing `EvaluationResult` remains the engine-level return type.
- YAML benchmark evaluators are unaffected in the first implementation pass.
- Future YAML support can add:

  ```yaml
  objective:
    entrypoint: predict
    direction: minimize
    metric: mae
    train_set:
      - {x: -4, answer: 30}
      - {x: 0, answer: 2}
  ```

## Resolved Decisions

1. **`train_set` rows**: plain mappings are accepted and coerced to
   `reps.Example`, but every row must still carry explicit input keys
   (`.with_inputs(...)`) — a row without them raises `ValueError` at
   `Objective` construction. Docs show `reps.Example`.
2. **Input-field inference**: not done. Explicit `.with_inputs(...)` is
   required, matching DSPy and avoiding silent label leakage.
3. **Minimize scoring**: `combined_score = -loss`; the raw loss is exposed
   under `best_metrics[metric_name]`. Per-instance scores use the same
   higher-is-better conversion (`-raw`); raw losses are echoed in `feedback`.
4. **`failure_score`**: the per-example metric value (natural space) used
   when an example's entrypoint/prediction/metric call raises, and applied
   to every example when the candidate fails to load. Default `0.0` suits
   maximize objectives; minimize objectives should pass a large positive
   value so crashing candidates are penalized.
