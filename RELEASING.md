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
for automated releases and `cargo publish` for distribution to crates.io.

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
5. The Release workflow re-runs the full CI suite against the tagged
   commit, then `cargo publish` ships the new version to crates.io.

Steps 4 and 5 are automatic. There is no manual `cargo publish` step.

### crates.io credential

The `publish` job in `.github/workflows/release.yml` needs a crates.io
API token in repo secret `CARGO_REGISTRY_TOKEN`.

- Create the token at https://crates.io/settings/tokens with:
  - **Scope:** `publish-update` only (no `publish-new` — the crate already exists).
  - **Crate scope:** `tkach` (limit to this crate).
  - **Expiry:** 1 year is reasonable.
- Store it: `gh secret set CARGO_REGISTRY_TOKEN -R kuchmenko/tkach -b "<token>"`,
  or via the GitHub UI: Settings → Secrets and variables → Actions →
  New repository secret.

### Why `cargo publish --dry-run` runs on every PR

`ci-checks.yml` includes a `package` job that runs `cargo publish --dry-run`
on every PR. This catches packaging issues (missing files, manifest
problems, doc-test fail under the published-crate build) **before** the
Release PR is merged — i.e., before release-please has cut a tag that
the real `cargo publish` would have to consume.

### Manual override

If you need to force a specific version or release out-of-band:

```
git tag tkach-vX.Y.Z
git push origin tkach-vX.Y.Z
```

The `tkach-` prefix matches release-please's tag format. After pushing,
publish manually with `cargo publish --token <token>` — the automated
publish workflow only fires on release-please-driven releases.
