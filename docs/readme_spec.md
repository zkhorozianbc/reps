# README Rewrite Spec

## 1. Goals

Rewrite `/home/user/reps/README.md` so the **Python API** (`reps.Optimizer().optimize(initial, evaluate)`) is the first install/usage flow a new visitor sees, while the existing CLI / YAML flow stays documented as the power-user path. Preserve the project's strongest signal — the Circle Packing n=26 benchmark headline — and surface the GEPA-inspired knobs (`selection_strategy`, `pareto_fraction`, `trace_reflection`, `merge`) that make REPS distinctive. Keep the rewrite tight: ~150 lines of README, no new prose claims beyond what's already shipped, and explicit cross-links to the design docs for users who want depth.

---

## 2. Proposed Top-Level Outline

In order, with a one-line intent each:

1. **Title + tagline + badges** — keep the existing layout (Circle Packing badge, Python 3.12 badge). One-sentence pitch line stays.
2. **Result: Circle Packing n=26** — keep verbatim. This is the headline; do not move below the fold.
3. **What REPS does** *(NEW)* — 3-5 bullets naming the load-bearing knobs (Pareto/MAP-Elites selection, trace reflection, system-aware merge, ancestry-aware reflection, convergence + SOTA steering). Each bullet maps to a constructor kwarg.
4. **Install** — keep existing `uv pip install -e .` block; add a one-line note that PyPI publish is pending and this README will be updated when it lands.
5. **Quick start (Python)** *(NEW, primary)* — the ~12-line minimum-viable example (see §4 below). Paragraph above it: "REPS is a library; pass an evaluator callable, get a result."
6. **What's an evaluator?** *(NEW)* — one paragraph answering the implicit question. `Callable[[str], float | dict | EvaluationResult]`. Show the three return shapes inline as a tiny code block.
7. **GEPA-style features (constructor knobs)** *(NEW)* — table of the load-bearing kwargs with a one-line description and default. Source of truth: `reps/api/optimizer.py:52`. Pointer to `docs/python_api_spec.md` for the full surface.
8. **Reusing a Model / advanced LLM config** *(NEW, brief)* — 6-line example showing `reps.Model(...)` for users who want to share one configured model across optimizers or call the model standalone.
9. **Power-user: CLI / YAML** *(MOVED, demoted)* — wraps the existing `reps-run`, "Add a Benchmark", and "Configs" sections under one H2. Lead-in sentence: "For batch experiments, reproducible runs, or YAML-driven sweeps, use the `reps-run` CLI."
   - 9.1 Run (`reps-run --config ...`) — keep existing content
   - 9.2 Add a benchmark — keep existing content
   - 9.3 Configs — keep existing content
10. **Tests** — keep verbatim (`uv run python -m pytest tests/`).
11. **Design docs** *(NEW)* — bulleted index of `docs/python_api_spec.md`, `docs/gepa_implementation_plan.md`, `docs/optimizer_engine_separation_spec.md` with one-line descriptions each.
12. **Acknowledgements / fork note** — promote the existing OpenEvolve attribution (currently buried in the tagline paragraph) into its own short footer section. One sentence.

Total target: ~150 README lines (today's is 124; net add ≈ 25-30 lines after demoting the CLI section and trimming).

---

## 3. Section-by-Section Delta

| # | Existing section (current line range) | Action | Notes |
|---|---|---|---|
| Title + badges (L1-8) | KEEP | No change. |
| Tagline para (L10) | REWRITE | Split: one sentence stays as the tagline; the OpenEvolve fork credit moves to §12 (Acknowledgements). |
| Result: Circle Packing (L12-24) | KEEP verbatim | Headline. Do not touch. |
| Install (L26-45) | KEEP, append one line | Add: "PyPI publish is in flight; this README will be updated to `pip install reps` (or equivalent) once the package is on PyPI. Until then, install from source as below." |
| Run (L47-64) | MOVE to §9.1 (Power-user: CLI / YAML) | Demoted under a new H2. Body unchanged. |
| Add a Benchmark (L66-108) | MOVE to §9.2 | Body unchanged. |
| Configs (L110-118) | MOVE to §9.3 | Body unchanged. |
| Tests (L120-124) | KEEP | No change. |

New sections to draft (sketch only — implementation subagent writes the prose):

### §3 What REPS does (NEW)

Sketch (bullets, not prose):

- **Adaptive selection** — `selection_strategy="map_elites" | "pareto" | "mixed"` with `pareto_fraction` for blending. (`reps/api/optimizer.py:52`, GEPA Phase 2)
- **Trace reflection** — `trace_reflection=True`: the reflection LLM sees per-instance scores + feedback, not just aggregate scores. (GEPA Phase 3)
- **Ancestry-aware reflection** — `lineage_depth=N`: reflection has access to the last N parents in a candidate's chain. (GEPA Phase 5)
- **System-aware merge** — `merge=True`: candidates from different islands recombine via LLM-driven merge prompt. (GEPA Phase 4)
- **Convergence + SOTA steering** — built-in convergence monitor (edit entropy + strategy divergence) and gap-aware compute steering when a target score is set. (Existing F4 + F6 — on by default.)

Each bullet ends with a parenthetical pointing at the file/line so curious readers can jump.

### §5 Quick start (Python) (NEW)

Lead-in: one sentence — "REPS is a Python library. Pass a seed program string and an evaluator callable; get back the best evolved program."

Then the 12-line example from §4 below.

### §6 What's an evaluator? (NEW)

One paragraph (sketch):

> An evaluator is any `Callable[[str], float | dict | reps.EvaluationResult]`. REPS calls it with the candidate program text and uses the returned score to drive selection. Return a `float` for a quick start, a `dict` with `combined_score` and optional `per_instance_scores` / `feedback` for richer signal, or a `reps.EvaluationResult` to unlock the per-objective Pareto + trace-reflection paths described in `docs/python_api_spec.md`.

Followed by a tiny three-shape code block:

```python
def eval_simple(code: str) -> float: return 1.0
def eval_dict(code: str) -> dict:    return {"combined_score": 0.9, "feedback": "..."}
def eval_full(code: str) -> reps.EvaluationResult: ...
```

### §7 GEPA-style features (NEW)

Two-column table — kwarg → one-line effect → default. Pulled from `reps/api/optimizer.py:170-195`. Sketch:

| Kwarg | Effect | Default |
|---|---|---|
| `selection_strategy` | `"map_elites"` (REPS classic), `"pareto"` (GEPA-style frontier), or `"mixed"` | `"map_elites"` |
| `pareto_fraction` | Blend ratio when `selection_strategy="mixed"` | `0.0` |
| `trace_reflection` | Reflection sees per-instance scores + feedback, not aggregates | `False` |
| `lineage_depth` | How many ancestors the reflection prompt sees | `3` |
| `merge` | Enable LLM-driven cross-island merge | `False` |
| `num_islands` | Population islands for diversity | `5` |
| `max_iterations` | Search budget | `100` |
| `output_dir` | Persist run artifacts; `None` ⇒ tempdir | `None` |

Footer line on the table: "Full surface (escape hatches, model knobs, deferred kwargs) in [docs/python_api_spec.md](docs/python_api_spec.md)."

### §8 Reusing a Model (NEW)

Sketch:

```python
import reps

model = reps.Model("anthropic/claude-sonnet-4.6", temperature=0.7)
print(model("hello"))                                    # standalone use

# Share one Model across multiple optimizers
o1 = reps.Optimizer(model=model, max_iterations=20)
o2 = reps.Optimizer(model=model, max_iterations=50, merge=True)
```

One sentence above it: "Most users pass a model-name string to `Optimizer(model=...)`. Build a `reps.Model` directly when you want to call the model outside the optimizer or share one configured client across multiple runs."

### §9 Power-user: CLI / YAML (MOVED)

H2 wrapper around the existing Run / Add a Benchmark / Configs sections (currently L47-118). Lead-in:

> For batch experiments, reproducible sweeps, or YAML-driven configuration, REPS ships a CLI: `reps-run --config <yaml>`. The Python API above is built on the same engine, so anything achievable via YAML is achievable via `Optimizer(...)` plus `Optimizer.from_config(cfg)`.

Then: existing `Run`, `Add a Benchmark`, `Configs` content as H3s (`### Run`, `### Add a benchmark`, `### Configs`).

### §11 Design docs (NEW)

Bulleted list (sketch):

- [`docs/python_api_spec.md`](docs/python_api_spec.md) — the v1 Python API contract: every public class, kwarg, return shape, with file:line references into the implementation.
- [`docs/gepa_implementation_plan.md`](docs/gepa_implementation_plan.md) — phase-by-phase rollout plan for the GEPA-inspired features (Pareto selection, trace reflection, merge, ancestry-aware reflection).
- [`docs/optimizer_engine_separation_spec.md`](docs/optimizer_engine_separation_spec.md) — internal refactor splitting the public `Optimizer` facade from the runtime engine.

### §12 Acknowledgements (NEW, footer)

One sentence (sketch): "Forked from [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve); now self-contained."

(This sentence currently lives at the end of L10. Move it to a footer so the lede stays clean.)

---

## 4. Minimum-Viable Python Example

Goes inside §5 ("Quick start (Python)"). Copy-paste-ready, 12 lines of executable code (excluding imports / blank lines):

```python
import reps

def evaluate(code: str) -> float:
    # Run the candidate, return a score. Higher is better.
    namespace = {}
    exec(code, namespace)
    return float(namespace["solve"]())

result = reps.Optimizer(
    model="anthropic/claude-sonnet-4.6",   # api_key from $ANTHROPIC_API_KEY
    max_iterations=20,
).optimize(
    initial=open("seed.py").read(),
    evaluate=evaluate,
)

print(result.best_score)
print(result.best_code)
```

Notes for the implementation subagent:
- Keep the `exec` evaluator deliberately trivial — the goal is "show the shape," not "show a real benchmark." A pointer to §9 ("Add a benchmark") covers the production path.
- The `# api_key from $ANTHROPIC_API_KEY` comment is load-bearing — it pre-empts "where do I put my key" without an extra paragraph.
- Do not show `extended_thinking=`, `selection_strategy=`, or `merge=` in this snippet. Those go in the §7 table. The quick-start must stay minimal.

---

## 5. Cross-Links

Every internal link the new README should include (with anchor where applicable):

| Link target | Where it appears | Purpose |
|---|---|---|
| `docs/python_api_spec.md` | §7 footer, §11 | Full Python API surface |
| `docs/gepa_implementation_plan.md` | §3 (parenthetically on Pareto/trace/merge bullets), §11 | GEPA phase rollout |
| `docs/optimizer_engine_separation_spec.md` | §11 | Engine/facade split |
| `experiment/configs/circle_sonnet_reps.yaml` | §9.1 (Run) | Existing reference config |
| `experiment/results/circle_sonnet_reps/packing.png` | §2 (Result) | Existing image, keep |
| `reps/config.py` | §9.3 (Configs) | Existing pointer to config schema |
| OpenEvolve repo URL | §12 (Acknowledgements) | Existing attribution |
| DeepMind validator Colab | §2 (Result) | Existing verification link |

External (kept from current README):
- `https://docs.astral.sh/uv/` (install section)
- DeepMind AlphaEvolve Colab (badge + verification)
- OpenEvolve GitHub (acknowledgements)

---

## 6. What NOT to Do

Explicit non-goals for the implementation subagent:

1. **Do not remove or move the Circle Packing n=26 result section.** It stays at §2, immediately after the badges. It is the project's strongest signal.
2. **Do not delete the CLI sections.** `reps-run`, "Add a benchmark", and "Configs" all stay — they move under §9 ("Power-user: CLI / YAML") but every existing line of content survives.
3. **Do not change the install command** to `pip install reps` or similar. PyPI publish is a separate spec; the README must continue to say `uv pip install -e .` until that ships. Add the one-line "PyPI publish pending" note and stop there.
4. **Do not invent kwargs.** Every constructor kwarg shown in the §7 table or the §4 quick-start must exist today in `reps/api/optimizer.py:52` (the `Optimizer.__init__` signature). If a kwarg is in `docs/python_api_spec.md` but marked deferred to v1.5, do not put it in the README.
5. **Do not show `from_config()` in the quick-start.** It belongs in §11 / `docs/python_api_spec.md` as an escape hatch. The README's primary flow is the simple constructor.
6. **Do not add a "Roadmap" or "Coming soon" section.** Deferred features live in `docs/python_api_spec.md` ("Deferred to v1.5+"). The README is for what works today.
7. **Do not expand the headline benchmark table.** It's the right size. Resist the urge to add more rows or a "How we did it" subsection.
8. **Do not add emojis** beyond the existing checkmark in the badge.
9. **Do not write new prose claims** about REPS's mechanism. Every descriptive sentence in §3 ("What REPS does") must be backed by an existing implementation file. If you can't cite a file:line, cut the bullet.
10. **Do not change the badge URLs.** The Circle Packing badge points to DeepMind's Colab on purpose — that's the third-party verification.

---

## Implementation checklist (for the next subagent)

1. Read this spec end-to-end.
2. Read the current `/home/user/reps/README.md`.
3. Apply the §3 deltas in order (top of file → bottom).
4. Verify every link in §5 resolves (paths exist on disk, URLs are unchanged from today's README).
5. Verify every kwarg shown is present in `reps/api/optimizer.py:52`.
6. Run a final pass: word count target ≈ 800-1000 words; line count target ≈ 150.
7. Do not add a "this README was rewritten on …" note. Just commit.
