"""`reps.Example` and `reps.Prediction` — DSPy-inspired data primitives.

`Example` is a dict-like record with named fields and explicit input keys.
`Prediction` is the same shape, used to wrap whatever a candidate program's
entrypoint returns so metric callables can compare `example.answer` to
`pred.answer` uniformly.

See docs/objective_api_spec.md.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping


class Example:
    """A dict-like data record with named fields and explicit input keys.

    Construct from keyword fields, a mapping, or any dict-like object (one
    exposing `keys()` + `__getitem__` — `dspy.Example` qualifies, so a DSPy
    dataset row drops straight in). Mark which fields are inputs with
    `.with_inputs(...)`. Supports both dot access (`ex.question`) and item
    access (`ex["question"]`). Field names that collide with a method
    (`inputs`, `labels`, `keys`, ...) are still reachable via `ex["inputs"]`.
    """

    def __init__(self, base: Mapping[str, Any] | None = None, **fields: Any) -> None:
        store: dict[str, Any] = {}
        if base is not None:
            if isinstance(base, Mapping):
                store.update(base)
            elif callable(getattr(base, "keys", None)) and hasattr(base, "__getitem__"):
                # Any dict-like object — covers dspy.Example, which exposes
                # keys() + __getitem__ but is not a collections.abc.Mapping.
                store.update({key: base[key] for key in base.keys()})
            else:
                raise TypeError(
                    f"reps.Example: `base` must be a mapping or a dict-like "
                    f"object (with keys() + __getitem__), got "
                    f"{type(base).__name__}"
                )
        store.update(fields)
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_input_keys", frozenset())

    # --- dict-like access ---------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        try:
            return self._store[key]
        except KeyError:
            raise KeyError(f"{type(self).__name__} has no field {key!r}") from None

    def __setitem__(self, key: str, value: Any) -> None:
        self._store[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def keys(self):
        return self._store.keys()

    def values(self):
        return self._store.values()

    def items(self):
        return self._store.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    # --- dot access ---------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when normal lookup fails. Underscore names are
        # never fields — raise so internal/copy/pickle access doesn't recurse.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._store[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__} has no field {name!r}"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._store[name] = value

    # --- input/label split --------------------------------------------------

    @property
    def input_keys(self) -> frozenset:
        return self._input_keys

    def with_inputs(self, *keys: str) -> "Example":
        """Return a copy with `keys` marked as the input fields."""
        missing = [k for k in keys if k not in self._store]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.with_inputs: unknown field(s) "
                f"{missing!r}; known fields are {sorted(self._store)!r}"
            )
        copy = type(self)(self._store)
        object.__setattr__(copy, "_input_keys", frozenset(keys))
        return copy

    def inputs(self) -> "Example":
        """Return a record containing only the input fields."""
        if not self._input_keys:
            raise ValueError(
                f"{type(self).__name__}.inputs(): no input keys set — call "
                f"`.with_inputs(...)` first."
            )
        sub = {k: v for k, v in self._store.items() if k in self._input_keys}
        copy = type(self)(sub)
        object.__setattr__(copy, "_input_keys", frozenset(self._input_keys))
        return copy

    def labels(self) -> "Example":
        """Return a record containing only the non-input fields."""
        if not self._input_keys:
            raise ValueError(
                f"{type(self).__name__}.labels(): no input keys set — call "
                f"`.with_inputs(...)` first."
            )
        sub = {k: v for k, v in self._store.items() if k not in self._input_keys}
        return type(self)(sub)

    # --- misc ---------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a shallow copy of the field store as a plain dict."""
        return dict(self._store)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Example):
            return NotImplemented
        return (
            type(self) is type(other)
            and self._store == other._store
            and self._input_keys == other._input_keys
        )

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self._store.items())
        if self._input_keys:
            return (
                f"{type(self).__name__}({fields}, "
                f"input_keys={sorted(self._input_keys)})"
            )
        return f"{type(self).__name__}({fields})"


class Prediction(Example):
    """Dict-like wrapper for a candidate program's entrypoint output.

    Same shape as `Example`. Metrics receive a `Prediction` so they can
    compare `example.answer` to `pred.answer` even when the candidate
    returned a bare scalar.
    """


def as_prediction(value: Any) -> Prediction:
    """Coerce a candidate entrypoint's return value into a `Prediction`.

    - `Prediction` -> returned unchanged.
    - mapping -> `Prediction(mapping)`.
    - object with `to_dict()` / `model_dump()` returning a mapping -> wrap it.
    - anything else (a scalar) -> `Prediction(answer=value)`.
    """
    if isinstance(value, Prediction):
        return value
    if isinstance(value, Mapping):
        return Prediction(dict(value))
    for attr in ("to_dict", "model_dump"):
        method = getattr(value, attr, None)
        if callable(method):
            mapping = method()
            if isinstance(mapping, Mapping):
                return Prediction(dict(mapping))
    return Prediction(answer=value)
