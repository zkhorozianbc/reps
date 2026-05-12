import json
from pathlib import Path

from reps.config import DatabaseConfig
from reps.database import Program, ProgramDatabase


def _database(tmp_path, **kwargs):
    cfg = DatabaseConfig(
        db_path=str(tmp_path / "db"),
        num_islands=1,
        feature_dimensions=["score"],
        **kwargs,
    )
    return ProgramDatabase(cfg)


def _program(program_id, score=1.0, code=None):
    return Program(
        id=program_id,
        code=code or f"def f(): return {score}",
        metrics={"combined_score": score},
    )


def test_save_prunes_stale_program_json_from_reused_directory(tmp_path):
    db = _database(tmp_path)
    db.add(_program("fresh", score=0.9))
    db.save()

    stale = Path(db.config.db_path) / "programs" / "stale.json"
    stale.write_text(
        json.dumps(
            {
                "id": "stale",
                "code": "def f(): return -1",
                "metrics": {"combined_score": 999.0},
            }
        )
    )

    db.save()

    loaded = _database(tmp_path)
    loaded.load(db.config.db_path)
    assert set(loaded.programs) == {"fresh"}
    assert not stale.exists()


def test_novelty_rejected_program_is_not_kept_in_database(tmp_path):
    db = _database(tmp_path)
    accepted = _program("accepted", score=0.5, code="def f(): return 1")
    accepted.embedding = [1.0, 0.0]
    db.add(accepted)

    db.embedding_client = type(
        "EmbeddingClient",
        (),
        {"get_embedding": lambda self, code: [1.0, 0.0]},
    )()
    db.similarity_threshold = 0.5
    db._llm_judge_novelty = lambda program, similar: False

    rejected = _program("rejected", score=0.6, code="def f(): return 1")
    db.add(rejected)

    assert "rejected" not in db.programs
    assert all("rejected" not in island for island in db.islands)
    assert db.best_program_id == "accepted"


def test_small_bytes_artifact_round_trips_as_bytes(tmp_path):
    db = _database(tmp_path)
    db.add(_program("p"))
    db.store_artifacts("p", {"payload.bin": b"\x00\x01hello"})

    assert db.get_artifacts("p") == {"payload.bin": b"\x00\x01hello"}


def test_large_artifacts_preserve_unsafe_and_colliding_names(tmp_path):
    db = _database(tmp_path, artifact_size_threshold=1)
    db.add(_program("p"))
    artifacts = {
        "../unsafe:name.txt": "alpha",
        "a/b.txt": "beta",
        "ab.txt": "gamma",
        "bytes/unsafe:name.txt": b"\xff\x00payload",
    }

    db.store_artifacts("p", artifacts)

    assert db.get_artifacts("p") == artifacts
