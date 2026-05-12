"""Tests for reps.utils — extracted utility functions."""

import math

import pytest

from reps.utils import (
    apply_diff,
    extract_diffs,
    format_metrics_safe,
    is_failed_evaluation_metrics,
    parse_full_rewrite,
    safe_numeric_average,
)


# ---------------------------------------------------------------------------
# is_failed_evaluation_metrics
# ---------------------------------------------------------------------------


class TestIsFailedEvaluationMetrics:
    def test_timeout_shape_is_failure(self):
        # Matches reps/evaluator.py:373 (asyncio.TimeoutError path).
        assert is_failed_evaluation_metrics({"error": 0.0, "timeout": True}) is True

    def test_cascade_stage1_timeout_shape_is_failure(self):
        # Matches reps/evaluator.py:539 (cascade stage1 timeout).
        assert (
            is_failed_evaluation_metrics(
                {"stage1_passed": 0.0, "error": 0.0, "timeout": True}
            )
            is True
        )

    def test_retries_exhausted_shape_is_failure(self):
        # Matches reps/evaluator.py:409 (all retries failed).
        assert is_failed_evaluation_metrics({"error": 0.0}) is True

    def test_normal_metrics_without_combined_score_is_not_failure(self):
        # Operator-misconfigured evaluator: real numbers but no combined_score.
        assert (
            is_failed_evaluation_metrics({"score": 0.5, "performance": 0.6}) is False
        )

    def test_normal_metrics_with_combined_score_is_not_failure(self):
        assert (
            is_failed_evaluation_metrics({"combined_score": 0.9, "score": 0.85})
            is False
        )

    def test_empty_dict_is_not_failure(self):
        # Empty dict isn't a sentinel — caller decides what to do.
        assert is_failed_evaluation_metrics({}) is False


# ---------------------------------------------------------------------------
# safe_numeric_average
# ---------------------------------------------------------------------------


class TestSafeNumericAverage:
    def test_basic_average(self):
        assert safe_numeric_average({"a": 2.0, "b": 4.0}) == 3.0

    def test_single_value(self):
        assert safe_numeric_average({"x": 7.0}) == 7.0

    def test_empty_dict(self):
        assert safe_numeric_average({}) == 0.0

    def test_ignores_non_numeric(self):
        result = safe_numeric_average({"a": 10.0, "b": "hello", "c": 20.0})
        assert result == 15.0

    def test_all_non_numeric(self):
        assert safe_numeric_average({"a": "x", "b": None}) == 0.0

    def test_ignores_nan(self):
        assert safe_numeric_average({"a": float("nan"), "b": 4.0}) == 4.0

    def test_integers(self):
        assert safe_numeric_average({"a": 3, "b": 7}) == 5.0


# ---------------------------------------------------------------------------
# extract_diffs
# ---------------------------------------------------------------------------


class TestExtractDiffs:
    def test_basic_extraction(self):
        diff_text = (
            "<<<<<<< SEARCH\n"
            "old line\n"
            "=======\n"
            "new line\n"
            ">>>>>>> REPLACE"
        )
        blocks = extract_diffs(diff_text)
        assert len(blocks) == 1
        assert blocks[0] == ("old line", "new line")

    def test_multiple_blocks(self):
        diff_text = (
            "<<<<<<< SEARCH\n"
            "first old\n"
            "=======\n"
            "first new\n"
            ">>>>>>> REPLACE\n"
            "some text\n"
            "<<<<<<< SEARCH\n"
            "second old\n"
            "=======\n"
            "second new\n"
            ">>>>>>> REPLACE"
        )
        blocks = extract_diffs(diff_text)
        assert len(blocks) == 2
        assert blocks[0] == ("first old", "first new")
        assert blocks[1] == ("second old", "second new")

    def test_no_blocks(self):
        assert extract_diffs("just plain text") == []


# ---------------------------------------------------------------------------
# apply_diff
# ---------------------------------------------------------------------------


class TestApplyDiff:
    def test_basic_application(self):
        original = "line1\nold line\nline3"
        diff_text = (
            "<<<<<<< SEARCH\n"
            "old line\n"
            "=======\n"
            "new line\n"
            ">>>>>>> REPLACE"
        )
        result = apply_diff(original, diff_text)
        assert result == "line1\nnew line\nline3"

    def test_no_match_leaves_unchanged(self):
        original = "line1\nline2\nline3"
        diff_text = (
            "<<<<<<< SEARCH\n"
            "nonexistent\n"
            "=======\n"
            "replacement\n"
            ">>>>>>> REPLACE"
        )
        result = apply_diff(original, diff_text)
        assert result == original

    def test_multiline_search_replace(self):
        original = "a\nb\nc\nd"
        diff_text = (
            "<<<<<<< SEARCH\n"
            "b\nc\n"
            "=======\n"
            "B\nC\n"
            ">>>>>>> REPLACE"
        )
        result = apply_diff(original, diff_text)
        assert result == "a\nB\nC\nd"


# ---------------------------------------------------------------------------
# parse_full_rewrite
# ---------------------------------------------------------------------------


class TestParseFullRewrite:
    def test_basic_code_extraction(self):
        response = "Here is the code:\n```python\nprint('hello')\n```\nDone."
        result = parse_full_rewrite(response)
        assert result == "print('hello')"

    def test_fallback_to_generic_block(self):
        response = "```\ngeneric code\n```"
        result = parse_full_rewrite(response)
        assert result == "generic code"

    def test_fallback_to_plain_text(self):
        response = "no code blocks here"
        result = parse_full_rewrite(response)
        assert result == "no code blocks here"

    def test_different_language(self):
        response = "```javascript\nconsole.log('hi')\n```"
        result = parse_full_rewrite(response, language="javascript")
        assert result == "console.log('hi')"


# ---------------------------------------------------------------------------
# format_metrics_safe
# ---------------------------------------------------------------------------


class TestFormatMetricsSafe:
    def test_formats_numeric(self):
        result = format_metrics_safe({"accuracy": 0.95})
        assert result == "accuracy=0.95000000"

    def test_formats_string_value(self):
        result = format_metrics_safe({"status": "ok"})
        assert result == "status=ok"

    def test_mixed_types(self):
        result = format_metrics_safe({"score": 1.5, "label": "good"})
        assert "score=1.50000000" in result
        assert "label=good" in result

    def test_empty_dict(self):
        assert format_metrics_safe({}) == ""

    def test_integer_value(self):
        result = format_metrics_safe({"count": 42})
        assert result == "count=42.00000000"
