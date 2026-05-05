"""Phase 6.2 adversarial / coverage-gap tests for `evaluate_with_promotion`.

Closes the categories the implementer covered only obliquely:

  (2) Legacy benchmark interaction — warning fires once across many
      iterations, names the benchmark file path, scratch evals via
      `evaluate_isolated` keep working when minibatch is set globally.
      Missing registry / `list_instances()` raises with a clear file path.
      `list_instances()` that itself raises propagates (does not silently
      fall back).
      `INSTANCES = []` collapses cleanly to full-eval (no zero-division
      or empty-call wedge).
  (3) Promotion threshold edge cases — boundary inclusivity (== threshold
      promotes), missing combined_score (treated as below threshold),
      NaN combined_score (NaN < threshold is False so currently promotes
      — pin so a future change is intentional), evaluator raising during
      minibatch (caught and tagged as low-fidelity), validity=0 with
      high score still promotes (only combined_score gates).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from reps.config import EvaluatorConfig
from reps.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_bench(
    tmp_path: Path,
    *,
    body: str,
    registry_kind: str = "constant",
    accepts_instances: bool = True,
    instances: list = None,
) -> Path:
    """Write a tiny benchmark module with a custom evaluate body.

    `body` is the function body (must include the `return` statement).
    """
    if instances is None:
        instances = ["t0", "t1", "t2", "t3"]
    if registry_kind == "constant":
        registry_block = f"INSTANCES = {instances!r}\n"
    elif registry_kind == "function":
        registry_block = (
            "def list_instances():\n"
            f"    return {instances!r}\n"
        )
    elif registry_kind == "function_raises":
        registry_block = (
            "def list_instances():\n"
            "    raise RuntimeError('registry boom')\n"
        )
    elif registry_kind == "missing":
        registry_block = ""
    else:
        raise AssertionError(f"unknown registry_kind {registry_kind!r}")

    if accepts_instances:
        sig = "def evaluate(program_path, env=None, instances=None):\n"
    else:
        sig = "def evaluate(program_path, env=None):\n"

    src = registry_block + "\n" + sig + body + "\n"
    bench = tmp_path / "bench.py"
    bench.write_text(src)
    return bench


def _make_evaluator(
    bench: Path,
    *,
    cascade: bool = False,
    minibatch_size=None,
    threshold: float = 0.5,
    strategy: str = "fixed_subset",
) -> Evaluator:
    cfg = EvaluatorConfig(
        cascade_evaluation=cascade,
        timeout=10,
        parallel_evaluations=1,
        max_retries=0,
        minibatch_size=minibatch_size,
        minibatch_promotion_threshold=threshold,
        minibatch_strategy=strategy,
    )
    return Evaluator(cfg, str(bench))


# ---------------------------------------------------------------------------
# Category 2: legacy benchmark / instance registry edge cases
# ---------------------------------------------------------------------------


class TestLegacyBenchmarkWarning:
    """The implementer reported a one-shot WARNING fallback. Pin:
        - emitted exactly once across many calls,
        - includes the benchmark file path so users can find it,
        - subsequent calls are silent and still produce full-eval results.
    """

    def test_warning_fires_exactly_once_across_many_iterations(
        self, tmp_path: Path, caplog
    ):
        body = (
            "    return {'combined_score': 0.7, 'validity': 1.0}\n"
        )
        bench = _write_bench(tmp_path, body=body, accepts_instances=False)
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)

        with caplog.at_level("WARNING"):
            for i in range(5):
                outcome = asyncio.run(
                    ev.evaluate_with_promotion(
                        "def f(): return 1\n", program_id=f"p{i}", iteration=i
                    )
                )
                assert outcome.metrics["fidelity"] == "full"

        # Exactly one warning total, regardless of how many iterations.
        warns = [
            r
            for r in caplog.records
            if r.levelname == "WARNING"
            and "instances" in r.getMessage().lower()
            and "minibatch_size" in r.getMessage()
        ]
        assert len(warns) == 1

    def test_warning_message_names_benchmark_file_path(self, tmp_path: Path, caplog):
        body = "    return {'combined_score': 0.7, 'validity': 1.0}\n"
        bench = _write_bench(tmp_path, body=body, accepts_instances=False)
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)

        with caplog.at_level("WARNING"):
            asyncio.run(
                ev.evaluate_with_promotion(
                    "def f(): return 1\n", program_id="p", iteration=0
                )
            )

        warns = [
            r for r in caplog.records if r.levelname == "WARNING"
            and "minibatch_size" in r.getMessage()
        ]
        assert len(warns) == 1
        # The warning must name the benchmark file path so the user can find it.
        assert str(bench) in warns[0].getMessage()


class TestInstanceRegistryEdgeCases:
    """Registry resolution failures must fail loud and name the file."""

    def test_missing_registry_error_names_benchmark_file(self, tmp_path: Path):
        body = "    return {'combined_score': 0.5, 'validity': 1.0}\n"
        bench = _write_bench(tmp_path, body=body, registry_kind="missing")
        ev = _make_evaluator(bench, minibatch_size=2)
        with pytest.raises(ValueError) as excinfo:
            asyncio.run(
                ev.evaluate_with_promotion(
                    "def f(): return 1\n", program_id="x", iteration=0
                )
            )
        msg = str(excinfo.value)
        # Must name the file and the missing API.
        assert str(bench) in msg
        assert "INSTANCES" in msg
        assert "list_instances" in msg

    def test_list_instances_raising_propagates(self, tmp_path: Path):
        # If the user's `list_instances()` itself raises, the harness
        # currently propagates the exception (does not silently fall back).
        # Pin this so a future refactor doesn't accidentally swallow it.
        body = "    return {'combined_score': 0.5, 'validity': 1.0}\n"
        bench = _write_bench(tmp_path, body=body, registry_kind="function_raises")
        ev = _make_evaluator(bench, minibatch_size=2)
        with pytest.raises(RuntimeError, match="registry boom"):
            asyncio.run(
                ev.evaluate_with_promotion(
                    "def f(): return 1\n", program_id="x", iteration=0
                )
            )

    def test_empty_instances_list_collapses_to_full_eval(self, tmp_path: Path):
        # `INSTANCES = []` must not wedge or zero-divide. The wiring's
        # empty-subset branch routes straight to full-eval.
        body = "    return {'combined_score': 0.7, 'validity': 1.0}\n"
        bench = _write_bench(tmp_path, body=body, instances=[])
        ev = _make_evaluator(bench, minibatch_size=2)
        outcome = asyncio.run(
            ev.evaluate_with_promotion(
                "def f(): return 1\n", program_id="x", iteration=0
            )
        )
        # No exception, falls through to full eval.
        assert outcome.metrics["fidelity"] == "full"
        assert outcome.metrics["combined_score"] == 0.7

    def test_registry_error_raised_per_call_not_at_construction(self, tmp_path: Path):
        # The implementer chose to raise lazily at first promotion call
        # rather than at evaluator construction. Pin this contract: an
        # `Evaluator(...)` for a missing-registry benchmark must succeed
        # construction (so legacy `evaluate_isolated` paths still work
        # for tool-call workers); only `evaluate_with_promotion` raises.
        body = "    return {'combined_score': 0.5, 'validity': 1.0}\n"
        bench = _write_bench(tmp_path, body=body, registry_kind="missing")
        # No raise here.
        ev = _make_evaluator(bench, minibatch_size=2)

        # Direct evaluate_isolated still works (scratch eval path that
        # tool-call workers use — minibatch promotion has no meaning there).
        outcome = asyncio.run(
            ev.evaluate_isolated("def f(): return 1\n", program_id="scratch")
        )
        assert outcome.metrics["combined_score"] == 0.5

        # Only evaluate_with_promotion raises.
        with pytest.raises(ValueError):
            asyncio.run(
                ev.evaluate_with_promotion(
                    "def f(): return 1\n", program_id="x", iteration=0
                )
            )


# ---------------------------------------------------------------------------
# Category 3: promotion threshold edge cases
# ---------------------------------------------------------------------------


class TestPromotionThresholdBoundary:
    """Pin boundary inclusivity. The implementer used `<` for rejection,
    so `mb_score == threshold` PROMOTES."""

    def test_score_equal_to_threshold_promotes(self, tmp_path: Path):
        # mb_score=0.5, threshold=0.5 -> not (0.5 < 0.5) -> promote.
        body = (
            "    if instances is not None:\n"
            "        return {'combined_score': 0.5, 'validity': 1.0}\n"
            "    return {'combined_score': 0.95, 'validity': 1.0}\n"
        )
        bench = _write_bench(tmp_path, body=body)
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        outcome = asyncio.run(
            ev.evaluate_with_promotion(
                "def f(): return 1\n", program_id="x", iteration=0
            )
        )
        assert outcome.metrics["fidelity"] == "full"
        assert outcome.metrics["combined_score"] == 0.95

    def test_score_just_below_threshold_rejects(self, tmp_path: Path):
        # epsilon below threshold -> reject.
        body = (
            "    if instances is not None:\n"
            "        return {'combined_score': 0.4999, 'validity': 1.0}\n"
            "    return {'combined_score': 0.95, 'validity': 1.0}\n"
        )
        bench = _write_bench(tmp_path, body=body)
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        outcome = asyncio.run(
            ev.evaluate_with_promotion(
                "def f(): return 1\n", program_id="x", iteration=0
            )
        )
        assert outcome.metrics["fidelity"] == "minibatch"


class TestPromotionMissingOrInvalidScore:
    """Defensive: minibatch eval with missing / non-numeric / NaN
    `combined_score` must not crash the harness."""

    def test_missing_combined_score_treated_as_below_threshold(self, tmp_path: Path):
        # `mb_score = mb_outcome.metrics.get("combined_score")` -> None;
        # `not isinstance(None, (int, float))` -> True -> reject.
        body = (
            "    if instances is not None:\n"
            "        return {'validity': 1.0}\n"  # no combined_score
            "    return {'combined_score': 0.95, 'validity': 1.0}\n"
        )
        bench = _write_bench(tmp_path, body=body)
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        outcome = asyncio.run(
            ev.evaluate_with_promotion(
                "def f(): return 1\n", program_id="x", iteration=0
            )
        )
        assert outcome.metrics["fidelity"] == "minibatch"
        # No combined_score in the rejected outcome — caller must handle this.
        assert "combined_score" not in outcome.metrics

    def test_string_combined_score_rejected(self, tmp_path: Path):
        # A misbehaving benchmark could return a non-numeric score.
        # `isinstance` check catches this before float() raises.
        body = (
            "    if instances is not None:\n"
            "        return {'combined_score': 'high', 'validity': 1.0}\n"
            "    return {'combined_score': 0.95, 'validity': 1.0}\n"
        )
        bench = _write_bench(tmp_path, body=body)
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        outcome = asyncio.run(
            ev.evaluate_with_promotion(
                "def f(): return 1\n", program_id="x", iteration=0
            )
        )
        assert outcome.metrics["fidelity"] == "minibatch"

    def test_nan_combined_score_currently_promotes(self, tmp_path: Path):
        # NaN < threshold is False, isinstance(NaN, float) is True. So
        # the implementer's gate `not isinstance(...) or float(x) < t`
        # evaluates to False and the program is PROMOTED.
        # This is a behavioral probe: pin the current contract so a
        # future change (e.g., math.isnan rejection) is intentional.
        body = (
            "    import math\n"
            "    if instances is not None:\n"
            "        return {'combined_score': float('nan'), 'validity': 1.0}\n"
            "    return {'combined_score': 0.95, 'validity': 1.0}\n"
        )
        bench = _write_bench(tmp_path, body=body)
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        outcome = asyncio.run(
            ev.evaluate_with_promotion(
                "def f(): return 1\n", program_id="x", iteration=0
            )
        )
        # Currently promotes. If this assertion flips to "minibatch" it
        # means the implementer added NaN guarding — update the doc.
        assert outcome.metrics["fidelity"] == "full"


class TestEvaluatorRaisesDuringMinibatch:
    """When the user's evaluator raises during the minibatch eval, the
    retry/error machinery returns an `{'error': 0.0}` metrics dict with
    no combined_score. The promotion gate sees that as "missing score"
    and rejects (tags fidelity=minibatch). Pin this so the failure mode
    stays predictable."""

    def test_evaluator_raises_during_minibatch_treated_as_low_fidelity(
        self, tmp_path: Path
    ):
        body = "    raise RuntimeError('benchmark crashed')\n"
        bench = _write_bench(tmp_path, body=body)
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        outcome = asyncio.run(
            ev.evaluate_with_promotion(
                "def f(): return 1\n", program_id="x", iteration=0
            )
        )
        # The retry pipeline catches the exception; mb_score is missing
        # from the resulting metrics dict, which rejects via the
        # `not isinstance(...)` branch. fidelity tagged as minibatch.
        assert outcome.metrics["fidelity"] == "minibatch"
        # The error metric is set so downstream loggers can see what happened.
        assert "error" in outcome.metrics


class TestValidityZeroOnMinibatch:
    """The promotion gate consults `combined_score` only — `validity` is
    not part of the gate. A program with validity=0 but combined_score
    above the threshold is promoted to full eval. Pin the contract so a
    future change (e.g., refusing promotion when validity == 0) is
    intentional."""

    def test_minibatch_high_score_low_validity_still_promotes(self, tmp_path: Path):
        body = (
            "    if instances is not None:\n"
            "        return {'combined_score': 0.95, 'validity': 0.0}\n"
            "    return {'combined_score': 0.95, 'validity': 1.0}\n"
        )
        bench = _write_bench(tmp_path, body=body)
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        outcome = asyncio.run(
            ev.evaluate_with_promotion(
                "def f(): return 1\n", program_id="x", iteration=0
            )
        )
        assert outcome.metrics["fidelity"] == "full"


# ---------------------------------------------------------------------------
# Cross-call invariants on the wiring
# ---------------------------------------------------------------------------


class TestPromotionWiringInvariants:
    """The fixed_subset invariant must be observable through the wiring,
    not just the helper. Two candidates evaluated at the same iteration
    must see the SAME instance subset."""

    def test_same_iteration_passes_same_subset_through_wiring(self, tmp_path: Path):
        # Bigger registry so size=3 picks a non-trivial subset.
        body = (
            "    import json\n"
            "    from pathlib import Path\n"
            "    p = Path(__file__).parent / 'calls.json'\n"
            "    calls = json.loads(p.read_text()) if p.exists() else []\n"
            "    calls.append(list(instances) if instances is not None else None)\n"
            "    p.write_text(json.dumps(calls))\n"
            "    return {'combined_score': 0.2, 'validity': 1.0}\n"
        )
        bench = _write_bench(
            tmp_path, body=body, instances=["a", "b", "c", "d", "e", "f", "g", "h"]
        )
        log = tmp_path / "calls.json"
        log.write_text("[]")

        ev = _make_evaluator(bench, minibatch_size=3, threshold=0.5)
        # Same iteration twice — different programs.
        asyncio.run(
            ev.evaluate_with_promotion(
                "def f(): return 1\n", program_id="a", iteration=2
            )
        )
        asyncio.run(
            ev.evaluate_with_promotion(
                "def g(): return 2\n", program_id="b", iteration=2
            )
        )
        calls = json.loads(log.read_text())
        # Both calls saw the SAME minibatch subset. (No full-eval calls
        # because both rejected at the threshold.)
        assert len(calls) == 2
        assert calls[0] is not None
        assert calls[1] is not None
        assert calls[0] == calls[1]

    def test_different_iterations_advance_window(self, tmp_path: Path):
        # iteration 0 and iteration `size` should yield different subsets
        # under fixed_subset. Pin through the wiring.
        body = (
            "    import json\n"
            "    from pathlib import Path\n"
            "    p = Path(__file__).parent / 'calls.json'\n"
            "    calls = json.loads(p.read_text()) if p.exists() else []\n"
            "    calls.append(list(instances) if instances is not None else None)\n"
            "    p.write_text(json.dumps(calls))\n"
            "    return {'combined_score': 0.2, 'validity': 1.0}\n"
        )
        bench = _write_bench(
            tmp_path, body=body, instances=["a", "b", "c", "d", "e", "f", "g", "h"]
        )
        log = tmp_path / "calls.json"
        log.write_text("[]")

        ev = _make_evaluator(bench, minibatch_size=3, threshold=0.5)
        asyncio.run(
            ev.evaluate_with_promotion(
                "def f(): return 1\n", program_id="a", iteration=0
            )
        )
        asyncio.run(
            ev.evaluate_with_promotion(
                "def g(): return 2\n", program_id="b", iteration=3
            )
        )
        calls = json.loads(log.read_text())
        assert calls[0] != calls[1]  # window advanced


class TestCascadeMutexAtRuntime:
    """If a user manages to construct a config with cascade=True via
    direct mutation (post-construction), `evaluate_with_promotion` falls
    through to the cascade path rather than running minibatch eval.
    Pin this so a future refactor doesn't accidentally try to combine
    them at runtime."""

    def test_cascade_path_taken_when_cascade_is_true_post_mutation(
        self, tmp_path: Path
    ):
        # Construct cleanly first (no minibatch_size) so __post_init__
        # passes; then enable both via mutation. evaluate_with_promotion
        # must defer to the cascade/direct path (here, direct since the
        # benchmark has no stage1/2/3) without raising.
        body = "    return {'combined_score': 0.7, 'validity': 1.0}\n"
        bench = _write_bench(tmp_path, body=body)
        cfg = EvaluatorConfig(
            cascade_evaluation=False,
            timeout=10,
            parallel_evaluations=1,
            max_retries=0,
            minibatch_size=2,
            minibatch_promotion_threshold=0.5,
        )
        # Sneak past the validator — simulate config drift.
        cfg.cascade_evaluation = True
        ev = Evaluator(cfg, str(bench))
        outcome = asyncio.run(
            ev.evaluate_with_promotion(
                "def f(): return 1\n", program_id="x", iteration=0
            )
        )
        # Falls through to the legacy path; tagged "full" not "minibatch".
        assert outcome.metrics["fidelity"] == "full"
