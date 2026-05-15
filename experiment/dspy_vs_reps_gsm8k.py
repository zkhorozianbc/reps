"""DSPy vs REPS bake-off on a small GSM8K slice.

Real public benchmark (grade-school math word problems via
`dspy.datasets.GSM8K`). Same model, same train/test slice, same
final-number exact-match scoring on both sides. Absolute accuracies on
the held-out test set are reported — no comparative-advantage framing,
just the numbers.

Cost: ~$1-3 of OpenRouter credits.
Time:  ~5-15 min wall-clock (LLM latency dominates).

Run:
    uv run python experiment/dspy_vs_reps_gsm8k.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

# Load .env so OPENROUTER_API_KEY is available to both REPS and DSPy.
from reps.runner import _load_dotenv

_load_dotenv(Path.cwd())
assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY not loaded"

import dspy  # noqa: E402
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric, parse_integer_answer  # noqa: E402
from dspy.teleprompt import BootstrapFewShot  # noqa: E402

import reps  # noqa: E402

MODEL_ID = "openrouter/anthropic/claude-sonnet-4.6"
N_TRAIN = 20
N_TEST = 20
REPS_ITERATIONS = 10


# --------------------------------------------------------------------------- #
# Shared metric — DSPy's own gsm8k_metric / parse_integer_answer, applied to
# both sides so the scoring is the canonical one used in DSPy's GSM8K runs.
# --------------------------------------------------------------------------- #


def numbers_match(gold: object, prediction: object) -> bool:
    """Match by `parse_integer_answer` (DSPy's GSM8K scorer).

    `only_first_line=False` so multi-line CoT outputs are scored on their
    final numerical token (the standard GSM8K convention). DSPy's structured
    `pred.answer` is a single line, so this is a no-op for the DSPy side.
    """
    return (
        int(parse_integer_answer(str(gold), only_first_line=False))
        == int(parse_integer_answer(str(prediction), only_first_line=False))
    )


# --------------------------------------------------------------------------- #
# Load GSM8K
# --------------------------------------------------------------------------- #


print(f"Loading GSM8K (train_size={N_TRAIN}, test_size={N_TEST}) ...", flush=True)
gsm = GSM8K()
train_examples = list(gsm.train[:N_TRAIN])
test_examples = list(gsm.test[:N_TEST])
print(f"  loaded {len(train_examples)} train + {len(test_examples)} test")
print(f"  sample Q: {train_examples[0].question[:90]}...")
print(f"  sample A: {train_examples[0].answer!r}")
print()


def gold_of(ex) -> str:
    """Fetch the gold answer field, regardless of whether `ex` is a
    dspy.Example or a reps.Example."""
    return ex["answer"] if "answer" in ex else getattr(ex, "answer")


def question_of(ex) -> str:
    return ex["question"] if "question" in ex else getattr(ex, "question")


def evaluate_on_test(predict_fn, label: str) -> dict:
    """Run `predict_fn(question_str) -> str` over the held-out test set."""
    correct = 0
    samples = []
    for ex in test_examples:
        q = question_of(ex)
        gold = gold_of(ex)
        try:
            pred = predict_fn(q)
        except Exception as exc:  # noqa: BLE001
            pred = f"<exception: {type(exc).__name__}: {exc}>"
        ok = numbers_match(gold, pred)
        correct += int(ok)
        if len(samples) < 3:
            samples.append(
                (q[:60], parse_integer_answer(str(gold)), parse_integer_answer(str(pred)), ok)
            )
    return {
        "label": label,
        "correct": correct,
        "total": len(test_examples),
        "accuracy": correct / len(test_examples) if test_examples else 0.0,
        "samples": samples,
    }


# --------------------------------------------------------------------------- #
# DSPy side
# --------------------------------------------------------------------------- #


print("=" * 60)
print("DSPy: dspy.ChainOfThought(question -> answer) + BootstrapFewShot")
print("=" * 60)

dspy.configure(
    lm=dspy.LM(MODEL_ID, api_key=os.environ["OPENROUTER_API_KEY"], temperature=0.0)
)


class GSMSignature(dspy.Signature):
    """Solve a grade-school math word problem. The answer should be the
    final numerical value (digits only)."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="final numerical answer, digits only")


student = dspy.ChainOfThought(GSMSignature)


optimizer = BootstrapFewShot(
    metric=gsm8k_metric,  # DSPy's canonical GSM8K scorer
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)

t0 = time.time()
compiled = optimizer.compile(student, trainset=train_examples)
dspy_compile_time = time.time() - t0
print(f"  BootstrapFewShot.compile completed in {dspy_compile_time:.1f}s", flush=True)


def dspy_predict(question: str) -> str:
    out = compiled(question=question)
    return out.answer


t0 = time.time()
dspy_result = evaluate_on_test(dspy_predict, "DSPy CoT+BootstrapFewShot")
dspy_eval_time = time.time() - t0
print(
    f"  DSPy test: {dspy_result['correct']}/{dspy_result['total']} = "
    f"{dspy_result['accuracy']:.1%} in {dspy_eval_time:.1f}s"
)
print(f"  samples: {dspy_result['samples']}")
print()


# --------------------------------------------------------------------------- #
# REPS side — general design: REPS evolves a PROMPT TEMPLATE directly.
# The seed is a natural-language string with a `{question}` placeholder;
# REPS' mutation worker (single_call SEARCH/REPLACE on the artifact text)
# improves the prompt. At evaluation each train example's `question`
# fills the placeholder, the LLM is called, and the metric scores the
# response. No `exec`, no `reps.runtime.llm` — `reps.PromptObjective`
# owns inference. This validates that the artifact-agnostic mutation
# templates produce a PROMPT (not Python) when handed a prompt seed.
# --------------------------------------------------------------------------- #


print("=" * 60)
print(f"REPS: PromptObjective.maximize(...), defaults, "
      f"{REPS_ITERATIONS} iterations")
print("=" * 60)


def reps_per_example_metric(example, pred, trace=None) -> float:
    return 1.0 if numbers_match(example.answer, pred.answer) else 0.0


# dspy.Example -> reps.Example via the dict-like protocol REPS accepts.
reps_train = [reps.Example(ex).with_inputs("question") for ex in train_examples]

reps_objective = reps.PromptObjective.maximize(
    train_set=reps_train,
    metric=reps_per_example_metric,
    model=MODEL_ID,
    parse=lambda out: str(parse_integer_answer(out, only_first_line=False)),
)

# Seed: a PROMPT TEMPLATE with a `{question}` placeholder — not Python.
# REPS evolves the prompt string itself (instructions, format, demos).
SEED = (
    "Solve the grade-school math problem below. Show your reasoning step by "
    "step, then output the final numerical answer on the last line.\n\n"
    "Question: {question}\n"
    "Answer:"
)

t0 = time.time()
reps_run = reps.Optimizer(
    model=MODEL_ID,
    max_iterations=REPS_ITERATIONS,
    num_islands=2,
).optimize(initial=SEED, objective=reps_objective, seed=42)
reps_compile_time = time.time() - t0
print(f"  REPS optimize completed in {reps_compile_time:.1f}s", flush=True)
print(f"  REPS best_score (train): {reps_run.best_score}")
print(f"  REPS best_metrics (train): {reps_run.best_metrics}")
print(f"  REPS total_tokens: {reps_run.total_tokens}")

# Build a predict_fn from the evolved PROMPT TEMPLATE.
_inference_lm = reps.Model(MODEL_ID, temperature=0.0)


def reps_predict(question: str) -> str:
    try:
        return _inference_lm(reps_run.best_code.format(question=question))
    except Exception as exc:  # noqa: BLE001
        return f"<exception: {type(exc).__name__}: {exc}>"


t0 = time.time()
reps_result = evaluate_on_test(reps_predict, "REPS PromptObjective")
reps_eval_time = time.time() - t0
print(
    f"  REPS test: {reps_result['correct']}/{reps_result['total']} = "
    f"{reps_result['accuracy']:.1%} in {reps_eval_time:.1f}s"
)
print(f"  samples: {reps_result['samples']}")
print()


# --------------------------------------------------------------------------- #
# Final tally
# --------------------------------------------------------------------------- #


print("=" * 60)
print(f"DSPy vs REPS — GSM8K ({N_TRAIN} train, {N_TEST} test)")
print("=" * 60)
print(f"  DSPy CoT+BootstrapFewShot : {dspy_result['accuracy']:.1%}  "
      f"({dspy_result['correct']}/{dspy_result['total']}, "
      f"compile {dspy_compile_time:.1f}s + eval {dspy_eval_time:.1f}s)")
print(f"  REPS PromptObjective.maximize(...) : {reps_result['accuracy']:.1%}  "
      f"({reps_result['correct']}/{reps_result['total']}, "
      f"optimize {reps_compile_time:.1f}s + eval {reps_eval_time:.1f}s)")
print()

# Sanity assertions: the artifact-agnostic templates must produce a PROMPT, not Python.
assert "{question}" in reps_run.best_code, (
    "Expected the evolved artifact to be a prompt template (still containing "
    f"{{question}}), got:\n{reps_run.best_code}"
)
assert "def " not in reps_run.best_code, (
    "Expected a prompt template; got something that looks like Python code:\n"
    f"{reps_run.best_code}"
)

print("--- REPS best_code ---")
print(reps_run.best_code)
