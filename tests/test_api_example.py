"""Unit tests for `reps.Example` and `reps.Prediction` (objective API layer)."""

from __future__ import annotations

import pytest

from reps.api.example import Example, Prediction, as_prediction


# --- construction -----------------------------------------------------------


def test_construct_from_kwargs():
    ex = Example(question="What is REPS?", answer="A program search harness.")
    assert ex.question == "What is REPS?"
    assert ex.answer == "A program search harness."


def test_construct_from_mapping():
    ex = Example({"x": 3, "answer": 2})
    assert ex.x == 3
    assert ex["answer"] == 2


def test_construct_from_mapping_plus_kwargs():
    ex = Example({"x": 3}, answer=2)
    assert ex.to_dict() == {"x": 3, "answer": 2}


def test_construct_rejects_non_mapping_base():
    with pytest.raises(TypeError, match="must be a mapping"):
        Example([("x", 1)])  # type: ignore[arg-type]


def test_construct_from_dict_like_object():
    """Accept any dict-like object exposing keys() + __getitem__ (e.g.
    dspy.Example, which is not a collections.abc.Mapping)."""

    class DictLike:
        def __init__(self, data):
            self._data = data

        def keys(self):
            return self._data.keys()

        def __getitem__(self, key):
            return self._data[key]

    ex = Example(DictLike({"x": 3, "answer": 2}))
    assert ex.to_dict() == {"x": 3, "answer": 2}


def test_construct_from_dict_like_object_plus_kwargs():
    class DictLike:
        def keys(self):
            return ["x"]

        def __getitem__(self, key):
            return {"x": 1}[key]

    ex = Example(DictLike(), answer=2)
    assert ex.to_dict() == {"x": 1, "answer": 2}


# --- access -----------------------------------------------------------------


def test_dot_and_item_access_agree():
    ex = Example(x=3, answer=2)
    assert ex.x == ex["x"] == 3


def test_missing_field_dot_access_raises_attribute_error():
    ex = Example(x=1)
    with pytest.raises(AttributeError, match="no field 'missing'"):
        _ = ex.missing


def test_missing_field_item_access_raises_key_error():
    ex = Example(x=1)
    with pytest.raises(KeyError, match="no field 'missing'"):
        _ = ex["missing"]


def test_contains_iter_len():
    ex = Example(x=1, answer=2)
    assert "x" in ex
    assert "missing" not in ex
    assert set(ex) == {"x", "answer"}
    assert len(ex) == 2


def test_item_assignment_updates_store():
    ex = Example(x=1)
    ex["answer"] = 9
    ex.note = "hi"
    assert ex.to_dict() == {"x": 1, "answer": 9, "note": "hi"}


def test_to_dict_returns_independent_copy():
    ex = Example(x=1)
    d = ex.to_dict()
    d["x"] = 999
    assert ex.x == 1


# --- with_inputs / inputs / labels -----------------------------------------


def test_with_inputs_marks_input_keys_and_copies():
    ex = Example(x=3, answer=2)
    marked = ex.with_inputs("x")
    assert marked.input_keys == frozenset({"x"})
    # original is untouched
    assert ex.input_keys == frozenset()
    # field data carried over
    assert marked.to_dict() == {"x": 3, "answer": 2}


def test_with_inputs_rejects_unknown_field():
    ex = Example(x=3, answer=2)
    with pytest.raises(ValueError, match="unknown field"):
        ex.with_inputs("nope")


def test_inputs_returns_only_input_fields():
    ex = Example(x=3, hint="h", answer=2).with_inputs("x", "hint")
    assert ex.inputs().to_dict() == {"x": 3, "hint": "h"}


def test_labels_returns_only_non_input_fields():
    ex = Example(x=3, hint="h", answer=2).with_inputs("x", "hint")
    assert ex.labels().to_dict() == {"answer": 2}


def test_inputs_without_input_keys_raises():
    ex = Example(x=3, answer=2)
    with pytest.raises(ValueError, match="no input keys set"):
        ex.inputs()


def test_labels_without_input_keys_raises():
    ex = Example(x=3, answer=2)
    with pytest.raises(ValueError, match="no input keys set"):
        ex.labels()


# --- equality / repr --------------------------------------------------------


def test_equality_considers_fields_and_input_keys():
    assert Example(x=1) == Example(x=1)
    assert Example(x=1) != Example(x=2)
    assert Example(x=1).with_inputs("x") != Example(x=1)
    assert Example(x=1).with_inputs("x") == Example(x=1).with_inputs("x")


def test_repr_is_readable():
    r = repr(Example(x=1).with_inputs("x"))
    assert "Example(" in r and "x=1" in r and "input_keys" in r


# --- Prediction + as_prediction --------------------------------------------


def test_prediction_is_example_subclass():
    assert issubclass(Prediction, Example)
    p = Prediction(answer=5)
    assert isinstance(p, Example)
    assert p.answer == 5


def test_as_prediction_passes_through_existing_prediction():
    p = Prediction(answer=1)
    assert as_prediction(p) is p


def test_as_prediction_wraps_scalar_as_answer():
    p = as_prediction(42)
    assert isinstance(p, Prediction)
    assert p.answer == 42


def test_as_prediction_wraps_mapping():
    p = as_prediction({"answer": "yes", "confidence": 0.9})
    assert isinstance(p, Prediction)
    assert p.answer == "yes"
    assert p["confidence"] == 0.9


def test_as_prediction_wraps_object_with_to_dict():
    class Boxed:
        def to_dict(self):
            return {"answer": 7}

    p = as_prediction(Boxed())
    assert p.answer == 7


def test_as_prediction_wraps_object_with_model_dump():
    class Pydanticish:
        def model_dump(self):
            return {"answer": "ok"}

    p = as_prediction(Pydanticish())
    assert p.answer == "ok"


def test_as_prediction_wraps_example_via_to_dict():
    ex = Example(answer=3, extra=1)
    p = as_prediction(ex)
    assert isinstance(p, Prediction)
    assert p.answer == 3
    assert p["extra"] == 1


def test_as_prediction_scalar_string_is_answer_not_mapping():
    p = as_prediction("a plain string")
    assert p.answer == "a plain string"
