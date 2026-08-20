# Releasing tkach

## Versioning

This project uses [Semantic Versioning](https://semver.org/) shaped by
release-please's pre-1.0 bumping flags:

- **Pre-1.0**: `0.MINOR.PATCH` — no stable API guarantees.
  - **Breaking** commits (`feat!:` / `fix!:`) bump **MINOR**.
  - **Everything else** that triggers a release (`feat:` / `fix:`) bumps **PATCH**.
- **Post-1.0**: standard SemVer.
  - `feat!:` / `fix!:` → MAJOR, `feat:` → MINOR, `fix:` → PATCH.

The pre-1.0 behaviour is configured by these `release-please-config.json` flags:

- `bump-minor-pre-major: true` — keeps breaking commits at MINOR (instead
  of MAJOR) while we're below 1.0.
- `bump-patch-for-minor-pre-major: true` — keeps non-breaking `feat:`
  commits at PATCH (instead of MINOR) while we're below 1.0.

The net effect: the MINOR digit signals "API broke", the PATCH digit
covers everything else — features, fixes, internals.

## How releases work

This project uses [release-please](https://github.com/googleapis/release-please)
for automated GitHub releases.

### Commit conventions

Commits must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
fix:   handle empty tool response  → bumps PATCH  (0.4.0 → 0.4.1)
feat:  add OpenAI provider         → bumps PATCH  (0.4.0 → 0.4.1) — pre-1.0
feat!: redesign Tool trait         → bumps MINOR  (0.4.0 → 0.5.0) — pre-1.0
chore: bump dependencies           → no release
docs:  update README               → no release
ci:    tighten clippy gate         → no release
refactor / test / perf             → no release
```

### Release flow

1. Push conventional commits to `main`.
2. release-please automatically opens or updates a **Release PR** with:
   - Version bump in `Cargo.toml`.
   - Updated `CHANGELOG.md`.
3. **Merge the Release PR** when meaningful changes have accumulated.
4. release-please cuts a GitHub Release + git tag from the merged commit.

Step 4 is automatic. The normal CI workflow runs independently on every
push to `main` and on every pull request.
