"""Phase 6.3 adversarial / coverage-gap tests for archive integrity.

Closes the categories the implementer covered only obliquely:

  (5) Archive integrity:
      - Minibatch-only programs reachable via diagnostics side dict.
      - All sample paths (sample, sample_from_island,
        sample_pareto_from_island) never return a minibatch-only program
        even when its combined_score / per_instance_scores would dominate.
      - Promoted programs aren't double-counted.
      - `db.save()` persists only archived programs (not the side dict).
      - `Program.metrics["fidelity"]` (a string in a Dict[str, float])
        round-trips through `to_dict`/`from_dict`.
      - last_iteration / iteration_found bookkeeping happens for the
        side-dict path too (so monitoring/diagnostics see the iteration).

  (6) `minibatch_archive_policy="all_with_tag"`:
      - Minibatch programs DO enter the archive and are visible to
        `sample()`.
      - The fidelity tag is preserved on the registered Program.
      - Selection treats them equally (no stratification yet — pin so a
        future stratification feature is intentional).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reps.config import DatabaseConfig
from reps.database import Program, ProgramDatabase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(
    policy: str = "promoted_only",
    num_islands: int = 2,
    db_path: str = None,
) -> ProgramDatabase:
    cfg = DatabaseConfig(
        in_memory=db_path is None,
        db_path=db_path,
        archive_size=10,
        population_size=100,
        num_islands=num_islands,
        feature_dimensions=["complexity", "diversity"],
        feature_bins=4,
    )
    return ProgramDatabase(cfg, minibatch_archive_policy=policy)


def _full_program(pid: str, score: float = 0.8) -> Program:
    return Program(
        id=pid,
        code=f"def f_{pid}(): return {score}\n",
        metrics={"combined_score": score, "fidelity": "full"},
    )


def _minibatch_program(pid: str, score: float = 0.2) -> Program:
    return Program(
        id=pid,
        code=f"def g_{pid}(): return {score}\n",
        metrics={"combined_score": score, "fidelity": "minibatch"},
    )


# ---------------------------------------------------------------------------
# Category 5: archive integrity edge cases
# ---------------------------------------------------------------------------


class TestMinibatchSideDictDiagnostics:
    """Spec: minibatch-only programs are 'still recorded for diagnostics'.
    Pin the access pattern so callers (review agent, debugging tools,
    metrics logger) can find them."""

    def test_side_dict_is_keyed_by_program_id(self):
        db = _make_db("promoted_only")
        db.add(_minibatch_program("mb_a", 0.1), iteration=2)
        db.add(_minibatch_program("mb_b", 0.3), iteration=3)
        # Direct dict access works.
        assert db._minibatch_only["mb_a"].metrics["fidelity"] == "minibatch"
        assert db._minibatch_only["mb_b"].metrics["combined_score"] == 0.3

    def test_side_dict_records_iteration_for_diagnostics(self):
        # Minibatch path still updates iteration_found and last_iteration
        # so diagnostics see when the rejection happened.
        db = _make_db("promoted_only")
        db.add(_minibatch_program("mb1", 0.2), iteration=7)
        prog = db._minibatch_only["mb1"]
        assert prog.iteration_found == 7
        assert db.last_iteration == 7

    def test_side_dict_overwrites_on_id_collision(self):
        # Two adds with the same id (e.g., re-evaluated program) — last
        # write wins. Same as the main programs dict semantics.
        db = _make_db("promoted_only")
        db.add(_minibatch_program("dup", 0.1), iteration=0)
        db.add(_minibatch_program("dup", 0.4), iteration=1)
        assert db._minibatch_only["dup"].metrics["combined_score"] == 0.4


class TestSampleNeverReturnsMinibatchOnly:
    """Concretely demonstrate that NONE of the sampling paths can return
    a minibatch-only program even when its scores would dominate."""

    def test_sample_does_not_return_minibatch_only(self):
        db = _make_db("promoted_only")
        db.add(_full_program("good", 0.5), iteration=0)
        # Add several minibatch-only programs with very high scores —
        # they would dominate IF they were considered.
        for i in range(5):
            db.add(_minibatch_program(f"mb_{i}", 0.99), iteration=1)
        for _ in range(30):
            parent, _ = db.sample()
            assert parent.id == "good"

    def test_sample_from_island_does_not_return_minibatch_only(self):
        db = _make_db("promoted_only")
        db.add(_full_program("good", 0.5), iteration=0)
        for i in range(5):
            db.add(_minibatch_program(f"mb_{i}", 0.99), iteration=1)
        for island_id in range(len(db.islands)):
            for _ in range(10):
                parent, _ = db.sample_from_island(island_id)
                assert parent.id == "good"

    def test_pareto_sampling_with_dominating_minibatch_program(self):
        # The classic Pareto risk: a minibatch-only program with
        # per_instance_scores that strictly dominate the only promoted
        # program. Under `promoted_only`, the sneaky program never
        # entered self.programs, so the Pareto frontier on the island
        # cannot include it.
        db = _make_db("promoted_only")
        promoted = _full_program("good", 0.5)
        promoted.per_instance_scores = {"a": 0.5, "b": 0.5, "c": 0.5}
        db.add(promoted, iteration=0)

        sneaky = _minibatch_program("sneaky", 0.99)
        sneaky.per_instance_scores = {"a": 0.99, "b": 0.99, "c": 0.99}
        db.add(sneaky, iteration=1)

        # The sneaky program is in _minibatch_only, NOT self.programs.
        assert "sneaky" in db._minibatch_only
        assert "sneaky" not in db.programs

        for _ in range(10):
            parent, _ = db.sample_pareto_from_island(0)
            assert parent.id == "good"

    def test_promoted_program_appears_in_only_one_dict(self):
        # Pin: a full-eval program enters self.programs but NOT _minibatch_only.
        db = _make_db("promoted_only")
        db.add(_full_program("p1", 0.7), iteration=0)
        assert "p1" in db.programs
        assert "p1" not in db._minibatch_only

    def test_minibatch_program_appears_in_only_one_dict(self):
        db = _make_db("promoted_only")
        db.add(_minibatch_program("m1", 0.1), iteration=0)
        assert "m1" in db._minibatch_only
        assert "m1" not in db.programs


class TestArchivePersistence:
    """Pin the on-disk save contract: only archived programs persist;
    the side dict is in-memory diagnostics only."""

    def test_save_persists_only_archived_programs(self, tmp_path: Path):
        db = _make_db("promoted_only", db_path=str(tmp_path))
        db.add(_full_program("kept", 0.7), iteration=0)
        db.add(_minibatch_program("rejected", 0.1), iteration=1)
        db.save()

        prog_dir = tmp_path / "programs"
        saved = sorted(p.name for p in prog_dir.iterdir())
        # Only the promoted program persists. The minibatch-only one is
        # deliberately omitted from on-disk state.
        assert saved == ["kept.json"]

    def test_save_metadata_does_not_reference_minibatch_only(self, tmp_path: Path):
        db = _make_db("promoted_only", db_path=str(tmp_path))
        db.add(_full_program("kept", 0.7), iteration=0)
        db.add(_minibatch_program("rejected", 0.1), iteration=1)
        db.save()

        with open(tmp_path / "metadata.json") as f:
            metadata = json.load(f)
        # No mention of the minibatch-only program in islands/archive.
        all_listed = set(metadata.get("archive", []))
        for island in metadata.get("islands", []):
            all_listed.update(island)
        assert "rejected" not in all_listed
        assert "kept" in all_listed


class TestProgramFidelityRoundTrip:
    """`Program.metrics` is annotated `Dict[str, float]` but the fidelity
    tag is a STRING. Pin that the string survives `to_dict`/`from_dict`
    so persisted minibatch programs (under all_with_tag) retain their
    tag after a load."""

    def test_fidelity_string_round_trips_through_to_dict_from_dict(self):
        original = Program(
            id="x",
            code="pass",
            metrics={"combined_score": 0.4, "fidelity": "minibatch"},
        )
        d = original.to_dict()
        # to_dict produces a plain dict; the fidelity string survives.
        assert d["metrics"]["fidelity"] == "minibatch"
        restored = Program.from_dict(d)
        assert restored.metrics["fidelity"] == "minibatch"
        assert isinstance(restored.metrics["fidelity"], str)

    def test_fidelity_full_round_trips(self):
        original = Program(
            id="y",
            code="pass",
            metrics={"combined_score": 0.9, "fidelity": "full"},
        )
        restored = Program.from_dict(original.to_dict())
        assert restored.metrics["fidelity"] == "full"

    def test_program_without_fidelity_round_trips(self):
        # Legacy: programs without a fidelity tag must still serialize.
        original = Program(
            id="legacy",
            code="pass",
            metrics={"combined_score": 0.5},
        )
        restored = Program.from_dict(original.to_dict())
        assert "fidelity" not in restored.metrics


class TestArchivePolicyFeatureMapAndIslandsUntouched:
    """Belt-and-suspenders: minibatch-only programs must not corrupt the
    MAP-Elites grid or per-island feature maps either, since the
    insertion path runs BEFORE feature calculation."""

    def test_minibatch_does_not_pollute_island_feature_maps(self):
        db = _make_db("promoted_only")
        # Empty before any add.
        assert all(len(m) == 0 for m in db.island_feature_maps)
        # Adding a minibatch-only program leaves feature maps untouched.
        db.add(_minibatch_program("m1", 0.1), iteration=0)
        assert all(len(m) == 0 for m in db.island_feature_maps)

    def test_minibatch_does_not_advance_archive(self):
        db = _make_db("promoted_only")
        for i in range(3):
            db.add(_minibatch_program(f"m{i}", 0.99), iteration=i)
        assert len(db.archive) == 0


# ---------------------------------------------------------------------------
# Category 6: all_with_tag policy
# ---------------------------------------------------------------------------


class TestAllWithTagSampleParity:
    """Under `all_with_tag`, minibatch programs are archived and
    visible to samplers — no stratification yet (deferred to v1.5).
    Pin the parity so a future change is intentional."""

    def test_all_with_tag_sample_can_return_minibatch_program(self):
        db = _make_db("all_with_tag")
        # Only program is a minibatch one.
        db.add(_minibatch_program("only_mb", 0.7), iteration=0)
        parent, _ = db.sample()
        assert parent.id == "only_mb"

    def test_all_with_tag_sample_from_island_returns_minibatch(self):
        db = _make_db("all_with_tag")
        db.add(_minibatch_program("only_mb", 0.7), iteration=0)
        # Find which island it landed in.
        landed_in = next(i for i, isl in enumerate(db.islands) if "only_mb" in isl)
        parent, _ = db.sample_from_island(landed_in)
        assert parent.id == "only_mb"

    def test_all_with_tag_pareto_can_return_minibatch_program(self):
        db = _make_db("all_with_tag")
        prog = _minibatch_program("only_mb", 0.7)
        prog.per_instance_scores = {"a": 0.7, "b": 0.7}
        db.add(prog, iteration=0)
        landed_in = next(i for i, isl in enumerate(db.islands) if "only_mb" in isl)
        parent, _ = db.sample_pareto_from_island(landed_in)
        assert parent.id == "only_mb"

    def test_all_with_tag_preserves_fidelity_after_add(self):
        db = _make_db("all_with_tag")
        prog = _minibatch_program("p1", 0.4)
        db.add(prog, iteration=0)
        # Tag preserved on the registered program (so future
        # stratification can read it).
        assert db.programs["p1"].metrics["fidelity"] == "minibatch"

    def test_all_with_tag_does_not_use_side_dict(self):
        # The side dict is exclusively for `promoted_only`. Under
        # `all_with_tag`, minibatch programs go straight into the main
        # archive, so the side dict stays empty.
        db = _make_db("all_with_tag")
        for i in range(3):
            db.add(_minibatch_program(f"mb_{i}", 0.4), iteration=i)
        assert len(db._minibatch_only) == 0
        assert all(f"mb_{i}" in db.programs for i in range(3))

    def test_all_with_tag_advances_archive_normally(self):
        # Under `all_with_tag` minibatch programs participate in the
        # archive update path and can become best/island-best.
        db = _make_db("all_with_tag")
        db.add(_minibatch_program("top", 0.9), iteration=0)
        # Archive update may or may not insert depending on capacity, but
        # best_program_id tracking always runs.
        assert db.best_program_id == "top"


class TestArchivePolicyDefaultsAndConstruction:
    """Pin the default and rejection of unknown policies at construction."""

    def test_default_policy_is_promoted_only(self):
        # The DatabaseConfig has no minibatch_archive_policy field — the
        # policy lives on EvaluatorConfig and is passed to the database
        # constructor. Pin that the constructor's default matches the
        # safe-by-default contract.
        cfg = DatabaseConfig(
            in_memory=True,
            archive_size=10,
            population_size=100,
            num_islands=2,
            feature_dimensions=["complexity", "diversity"],
            feature_bins=4,
        )
        db = ProgramDatabase(cfg)  # no policy kwarg
        assert db.minibatch_archive_policy == "promoted_only"

    def test_unknown_policy_rejected_at_construction_with_named_field(self):
        cfg = DatabaseConfig(
            in_memory=True,
            archive_size=10,
            population_size=100,
            num_islands=2,
            feature_dimensions=["complexity", "diversity"],
            feature_bins=4,
        )
        with pytest.raises(ValueError) as excinfo:
            ProgramDatabase(cfg, minibatch_archive_policy="bogus_policy")
        msg = str(excinfo.value)
        # Should name the bad value so user can find their typo.
        assert "bogus_policy" in msg
        # Lists at least one valid value.
        assert "promoted_only" in msg or "all_with_tag" in msg
