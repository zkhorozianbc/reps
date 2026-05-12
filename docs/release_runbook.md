# Release Runbook

Per-release procedure for the `reps-py` package on PyPI.

## One-time bootstrap (do these first, exactly once)

These three steps must complete before the automated `release.yml`
workflow can publish. After they're done, every subsequent release is
just a tag push.

### 1. Manually upload `0.1.0` to PyPI

Claim the `reps-py` namespace from a maintainer's laptop. PyPI's
Trusted Publishing requires the project to already exist before it
can be configured, so the first release MUST be a manual upload.

Prerequisites: a PyPI account, and the maintainer is a Project Owner
(or Project Maintainer with upload permission) on `reps-py` once
created.

```bash
cd /path/to/reps                      # repo root
uv build                              # produces dist/reps_search-0.1.0-*
uv publish                            # prompts for PyPI credentials
                                      # OR: PYPI_TOKEN=pypi-... uv publish
```

`uv publish` reads `~/.pypirc` or `PYPI_TOKEN`/`UV_PUBLISH_TOKEN` from
the env. See [uv publish docs](https://docs.astral.sh/uv/reference/cli/#uv-publish).

Verify: visit `https://pypi.org/project/reps-py/0.1.0/` and
confirm the wheel + sdist are listed.

### 2. Configure Trusted Publishing on PyPI

Trusted Publishing lets the GitHub Actions workflow obtain a
short-lived credential via OIDC instead of storing a long-lived API
token. This is what `release.yml` already expects.

1. Open `https://pypi.org/manage/project/reps-py/settings/publishing/`.
2. Under **Add a new pending publisher**, create one with:
   - **Owner:** `zkhorozianbc`
   - **Repository name:** `reps`
   - **Workflow name:** `release.yml`
   - **Environment name:** `pypi`
3. Save.

After this, any push of a `v*.*.*` tag will trigger `release.yml`,
which will be permitted to publish.

### 3. Create the `pypi` GitHub environment

Trusted Publishing also requires the workflow to declare an
environment. The `release.yml` already references `environment: name:
pypi`; this step creates that environment in the repo so the workflow
can run.

1. Open `https://github.com/zkhorozianbc/reps/settings/environments`.
2. Click **New environment**, name it `pypi`.
3. (Optional but recommended) Add a **Required reviewer** = the
   maintainer's GitHub user. This makes every release require a
   one-click approval before publish, useful as a safety gate.
4. Save.

After all three bootstrap steps complete, the project is ready for
automated releases.

## Per-release procedure (every time after bootstrap)

For each new release (patch, minor, or major — see version policy in
[`docs/release_spec.md`](./release_spec.md#sub-spec-1-version-bump-policy)):

### 1. Update `CHANGELOG.md`

Move items from `[Unreleased]` into a new `[X.Y.Z] - YYYY-MM-DD`
section. Add a fresh empty `[Unreleased]` block above it.

```markdown
## [Unreleased]

### Added
### Changed
### Fixed

## [0.1.1] - 2026-05-15

### Fixed
- Brief description of the fix.
```

### 2. Bump the version in `pyproject.toml`

Edit `[project] version = "X.Y.Z"`. The pre-1.0 bump rules:

| Change                              | Bump  |
|-------------------------------------|-------|
| Bug fix, docs, internal refactor    | patch |
| ANY public API change (additive or breaking) | minor |
| Graduation to stable API            | 1.0.0 (one-time) |

Post-1.0 switches to strict semver. Worked examples for past commits
in [`docs/release_spec.md`](./release_spec.md#22-reference-table--actual-reps-changes-so-far).

### 3. Commit, tag, push

```bash
git add CHANGELOG.md pyproject.toml
git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main
git push origin vX.Y.Z
```

### 4. Watch the workflow

Open `https://github.com/zkhorozianbc/reps/actions`. The
`release.yml` workflow runs in ~2 minutes and:

1. Checks the tag matches `pyproject.toml`'s version (fails fast if
   not).
2. Runs `uv build` to produce sdist + wheel.
3. Publishes to PyPI via OIDC Trusted Publishing.

If you set up the `pypi` environment with a required reviewer, the
publish step blocks on your approval. Click through to approve.

### 5. Verify

```bash
uv pip install reps-py==X.Y.Z   # in a clean env
python -c "import reps; print(reps.__version__ if hasattr(reps, '__version__') else 'ok')"
```

Visit `https://pypi.org/project/reps-py/X.Y.Z/` to confirm.

## Hotfix procedure

For urgent fixes (e.g., a broken release):

1. Branch from the broken tag: `git checkout -b hotfix/X.Y.Z+1 vX.Y.Z`.
2. Commit the fix; bump version to `X.Y.Z+1` in `pyproject.toml`.
3. Cherry-pick the fix forward into `main` after release if needed.
4. Tag + push as in the per-release procedure.

For a *yanked* release (broken enough to not run): use PyPI's web UI
at `https://pypi.org/manage/project/reps-py/release/X.Y.Z/` to
yank. Yanked releases stay installable by exact-pinned consumers but
disappear from `pip install reps-py` resolution.

## Rollback procedure

PyPI does **not** allow re-uploading a deleted version (the version
number is permanently consumed). Options when a bad release lands:

1. **Yank** (recommended) — stops new resolutions, keeps existing
   pins working.
2. **Release X.Y.Z+1 immediately** — fix-forward.

Avoid `pip install --force-reinstall` chains; ship a new version
instead.

## Common pitfalls

- **Tag pushed without committing version bump.** The `release.yml`
  validates `tag == pyproject.toml.version` and fails. Fix: delete
  the tag (`git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`),
  amend the commit, retag, repush.
- **Trusted Publishing not configured.** First tag push fails with
  "no trusted publisher matches". Re-run step 2 of the bootstrap.
- **`pypi` environment missing.** Workflow log shows "environment
  'pypi' does not exist". Re-run step 3 of the bootstrap.
- **Version regression.** `0.2.0 → 0.1.5` is rejected by PyPI.
  Always bump forward.
- **`tqdm` / `python-dotenv` / `flask` / `bs4` referenced in new
  code.** These were dropped from deps in `4c2a36f`. If new code
  imports them, add them back to `dependencies` first or you'll
  break a clean install.

## References

- [`docs/release_spec.md`](./release_spec.md) — full version policy +
  workflow YAML.
- [`docs/packaging_spec.md`](./packaging_spec.md) — package
  metadata + dep extras.
- [`CHANGELOG.md`](../CHANGELOG.md) — release history.
- [`pyproject.toml`](../pyproject.toml) — version + metadata source
  of truth.
- [PyPI Trusted Publishing
  docs](https://docs.pypi.org/trusted-publishers/).
- [`uv publish`
  reference](https://docs.astral.sh/uv/reference/cli/#uv-publish).
