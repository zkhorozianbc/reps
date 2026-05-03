"""Contract tests for the GEPA-style ASI extension fields (Phase 1.1).

Validates that:
- EvaluationResult accepts and exposes per_instance_scores + feedback (optional)
- EvaluationOutcome carries them through
- Program persists them via to_dict/from_dict round-trip
- Old serialized dicts without these fields still deserialize (backward compat)
- Defaults are None when omitted
"""

import json
from pathlib import Path

import pytest

from reps.database import Program
from reps.evaluation_result import EvaluationResult
from reps.evaluator import EvaluationOutcome


class TestEvaluationResultFields:
    def test_defaults_when_omitted(self):
        r = EvaluationResult(metrics={"combined_score": 0.5})
        assert r.per_instance_scores is None
        assert r.feedback is None
        assert r.artifacts == {}

    def test_carries_per_instance_scores(self):
        r = EvaluationResult(
            metrics={"combined_score": 0.7},
            per_instance_scores={"task_a": 0.9, "task_b": 0.5},
        )
        assert r.per_instance_scores == {"task_a": 0.9, "task_b": 0.5}

    def test_carries_feedback(self):
        r = EvaluationResult(
            metrics={"combined_score": 0.0},
            feedback="task_a: TypeError on input 3; task_b: ok",
        )
        assert r.feedback == "task_a: TypeError on input 3; task_b: ok"

    def test_from_dict_preserves_metrics_only(self):
        r = EvaluationResult.from_dict({"combined_score": 0.5, "validity": 1.0})
        assert r.metrics == {"combined_score": 0.5, "validity": 1.0}
        assert r.per_instance_scores is None
        assert r.feedback is None

    def test_from_dict_peels_top_level_per_instance_scores(self):
        """When a user's evaluator returns a dict with `per_instance_scores`
        at the top level, from_dict must extract it into the dedicated
        field — not leave it inside `metrics` where the controller can't
        see it (and where it'd silently break GEPA Phases 2-5)."""
        r = EvaluationResult.from_dict({
            "combined_score": 0.5,
            "validity": 1.0,
            "per_instance_scores": {"task_a": 0.7, "task_b": 0.3},
        })
        assert r.per_instance_scores == {"task_a": 0.7, "task_b": 0.3}
        # And NOT smuggled inside metrics.
        assert "per_instance_scores" not in r.metrics
        # Other metrics are untouched.
        assert r.metrics == {"combined_score": 0.5, "validity": 1.0}

    def test_from_dict_peels_top_level_feedback(self):
        r = EvaluationResult.from_dict({
            "combined_score": 0.0,
            "feedback": "task_a failed: TypeError",
        })
        assert r.feedback == "task_a failed: TypeError"
        assert "feedback" not in r.metrics
        assert r.metrics == {"combined_score": 0.0}

    def test_from_dict_peels_both_when_present(self):
        r = EvaluationResult.from_dict({
            "combined_score": 0.4,
            "per_instance_scores": {"a": 0.8, "b": 0.0},
            "feedback": "b regressed",
        })
        assert r.per_instance_scores == {"a": 0.8, "b": 0.0}
        assert r.feedback == "b regressed"
        assert r.metrics == {"combined_score": 0.4}

    def test_from_dict_does_not_mutate_input(self):
        """Caller's dict must not be mutated — they may keep using it
        after passing it in."""
        original = {
            "combined_score": 0.4,
            "per_instance_scores": {"a": 0.8},
            "feedback": "msg",
        }
        before = dict(original)
        EvaluationResult.from_dict(original)
        assert original == before


class TestEvaluationOutcomeFields:
    def test_defaults_when_omitted(self):
        o = EvaluationOutcome(metrics={"combined_score": 0.5})
        assert o.per_instance_scores is None
        assert o.feedback is None

    def test_carries_new_fields(self):
        o = EvaluationOutcome(
            metrics={"combined_score": 0.7},
            program_id="abc",
            per_instance_scores={"i0": 1.0, "i1": 0.0},
            feedback="i1 failed: off-by-one",
        )
        assert o.per_instance_scores == {"i0": 1.0, "i1": 0.0}
        assert o.feedback == "i1 failed: off-by-one"


class TestProgramFields:
    def test_defaults_when_omitted(self):
        p = Program(id="x", code="pass")
        assert p.per_instance_scores is None
        assert p.feedback is None

    def test_carries_new_fields(self):
        p = Program(
            id="x",
            code="pass",
            metrics={"combined_score": 0.4},
            per_instance_scores={"task_a": 0.6, "task_b": 0.2},
            feedback="task_b regressed vs parent",
        )
        assert p.per_instance_scores == {"task_a": 0.6, "task_b": 0.2}
        assert p.feedback == "task_b regressed vs parent"

    def test_round_trip_preserves_new_fields(self):
        p = Program(
            id="x",
            code="pass",
            metrics={"combined_score": 0.4},
            per_instance_scores={"task_a": 0.6},
            feedback="ok",
        )
        d = p.to_dict()
        assert d["per_instance_scores"] == {"task_a": 0.6}
        assert d["feedback"] == "ok"
        p2 = Program.from_dict(d)
        assert p2.per_instance_scores == {"task_a": 0.6}
        assert p2.feedback == "ok"

    def test_round_trip_preserves_none_defaults(self):
        p = Program(id="x", code="pass", metrics={"combined_score": 0.4})
        p2 = Program.from_dict(p.to_dict())
        assert p2.per_instance_scores is None
        assert p2.feedback is None

    def test_backward_compat_old_dict_without_new_fields(self):
        # Simulate a Program saved before Phase 1 — no per_instance_scores/feedback keys.
        legacy = {
            "id": "old",
            "code": "pass",
            "changes_description": "seed",
            "language": "python",
            "parent_id": None,
            "generation": 0,
            "timestamp": 0.0,
            "iteration_found": 0,
            "metrics": {"combined_score": 0.42},
            "complexity": 0.0,
            "diversity": 0.0,
            "metadata": {},
            "prompts": None,
            "artifacts_json": None,
            "artifact_dir": None,
            "embedding": None,
        }
        p = Program.from_dict(legacy)
        assert p.id == "old"
        assert p.metrics == {"combined_score": 0.42}
        assert p.per_instance_scores is None
        assert p.feedback is None

    def test_copy_preserves_new_fields_independent_of_source(self):
        # Mirrors the database.py island/migration copy idiom: build a child
        # Program from a parent and confirm both fields propagate, with
        # per_instance_scores deep-copied so mutating one doesn't affect the other.
        parent = Program(
            id="parent",
            code="pass",
            metrics={"combined_score": 0.5},
            per_instance_scores={"i0": 0.7},
            feedback="parent feedback",
        )
        copy = Program(
            id="copy",
            code=parent.code,
            metrics=parent.metrics.copy(),
            per_instance_scores=(
                dict(parent.per_instance_scores)
                if parent.per_instance_scores is not None
                else None
            ),
            feedback=parent.feedback,
        )
        assert copy.per_instance_scores == {"i0": 0.7}
        assert copy.feedback == "parent feedback"
        # Mutate the copy's per_instance_scores and confirm parent unchanged.
        copy.per_instance_scores["i0"] = 0.0
        assert parent.per_instance_scores == {"i0": 0.7}

    def test_copy_preserves_none_when_source_has_no_new_fields(self):
        parent = Program(id="p", code="pass", metrics={"combined_score": 0.5})
        copy = Program(
            id="c",
            code=parent.code,
            metrics=parent.metrics.copy(),
            per_instance_scores=(
                dict(parent.per_instance_scores)
                if parent.per_instance_scores is not None
                else None
            ),
            feedback=parent.feedback,
        )
        assert copy.per_instance_scores is None
        assert copy.feedback is None

    def test_json_round_trip_through_disk(self, tmp_path: Path):
        p = Program(
            id="disk",
            code="pass",
            metrics={"combined_score": 0.4},
            per_instance_scores={"i0": 1.0, "i1": 0.5},
            feedback="i1 partial",
        )
        path = tmp_path / "p.json"
        path.write_text(json.dumps(p.to_dict()))
        loaded = Program.from_dict(json.loads(path.read_text()))
        assert loaded.per_instance_scores == {"i0": 1.0, "i1": 0.5}
        assert loaded.feedback == "i1 partial"


class TestCascadeAsiPreservation:
    """Cascade evaluation merges multiple stages — ASI fields from stage2/3
    must survive the merge step, otherwise they never reach the Program."""

    def test_stage2_asi_survives_merge(self, tmp_path: Path):
        import asyncio
        from reps.config import EvaluatorConfig
        from reps.evaluator import Evaluator

        # Tiny inline benchmark: stage1 returns metrics dict only;
        # evaluate (stage2) returns full EvaluationResult with ASI.
        bench = tmp_path / "bench_asi.py"
        bench.write_text(
            "from reps.evaluation_result import EvaluationResult\n"
            "def evaluate_stage1(program_path, **kw):\n"
            "    return {'combined_score': 0.6, 'validity': 1.0}\n"
            "def evaluate(program_path, **kw):\n"
            "    return EvaluationResult(\n"
            "        metrics={'combined_score': 0.6, 'validity': 1.0},\n"
            "        per_instance_scores={'task_a': 0.7, 'task_b': 0.5},\n"
            "        feedback='task_b regressed',\n"
            "    )\n"
            "def evaluate_stage2(program_path, **kw):\n"
            "    return evaluate(program_path)\n"
        )
        prog = tmp_path / "prog.py"
        prog.write_text("def f(): return 1\n")

        cfg = EvaluatorConfig(
            cascade_evaluation=True,
            cascade_thresholds=[0.0, 0.0],
            timeout=10,
            parallel_evaluations=1,
            max_retries=0,
        )
        evaluator = Evaluator(cfg, str(bench))
        outcome = asyncio.run(
            evaluator.evaluate_isolated(prog.read_text(), program_id="t")
        )
        assert outcome.per_instance_scores == {"task_a": 0.7, "task_b": 0.5}
        assert outcome.feedback == "task_b regressed"
