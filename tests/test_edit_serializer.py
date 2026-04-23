from reps.config import Config
from reps.utils import extract_diffs
from reps.workers.edit_serializer import serialize_diff_blocks


def test_serialize_roundtrips_through_extract_diffs():
    # extract_diffs calls .rstrip() on both sides, so blocks must not have
    # trailing whitespace for the roundtrip to be exact.
    blocks = [
        ("old_a", "new_a"),
        ("def foo():\n    pass", "def foo():\n    return 1"),
    ]
    text = serialize_diff_blocks(blocks)
    pattern = Config().diff_pattern
    extracted = extract_diffs(text, pattern)
    assert extracted == blocks


def test_serialize_handles_missing_trailing_newlines():
    blocks = [("search_no_nl", "replace_no_nl")]
    text = serialize_diff_blocks(blocks)
    assert "<<<<<<< SEARCH" in text
    assert "=======" in text
    assert ">>>>>>> REPLACE" in text
