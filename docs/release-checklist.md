# Release Checklist — v0.1.0

Run every step on a clean checkout at the release tag.

> **Dry-run status — 2026-11-03.** Sections 1, 2 (doctor/demo), 4, and 5 pass
> on the release commit. Signing and tagging stay unchecked until a valid
> release key is available.

## 1. Freeze contracts

- [ ] `docs/rfcs/0001-protocol-v1.md` matches `src/lmpu/schema/protocol-v1.schema.json`.
- [ ] `make schema` produces no diff.
- [ ] `tests/unit/test_schema.py` passes.
- [ ] Trace schema is unchanged or additive-only.

## 2. Verify the product surface

- [ ] `uv run lmpu demo canonical` prints three `PASS` lines.
- [ ] `uv run lmpu doctor` passes.
- [ ] `uv run lmpu trace latest --gaps` exits `0` on a complete trace.
- [ ] Dashboard loads `Parked Host`, `Jobs`, `Buffers`, and `Live Events`.
- [ ] `uv run lmpu plugin hello` creates an importable plugin.
- [ ] `docs/quickstart.md` commands work verbatim.

## 3. Test gates

- [ ] `make check`
- [ ] `make test-core`
- [ ] `make test-gpu`
- [ ] `make test-failure`
- [ ] `make test-security`
- [ ] `make bench` publishes a report with C1–C5 statuses and limitations.

## 4. Build and inspect

- [ ] `make release-build`
- [ ] `uv build` produces `lmpu-0.1.0-py3-none-any.whl` and `lmpu-0.1.0.tar.gz`.
- [ ] Wheel contains `lmpu/schema/protocol-v1.schema.json` and `trace-v1.schema.json`.
- [ ] Install the wheel in a clean venv and run `lmpu version`.
- [ ] `dist/SHA256SUMS` matches the artifacts.

## 5. SBOM and audit

- [ ] `make release-sbom` writes `dist/sbom.json`.
- [ ] `uvx pip-audit` reports no known vulnerabilities, or each finding has a
      documented exception.

## 6. Security notes

- [ ] `SECURITY.md` describes sandbox boundaries, socket permissions, redaction,
      and the supported-version policy.
- [ ] No secrets, tokens, or personal paths are in the wheel, sdist, traces, or
      benchmark artifacts.

## 7. Sign and tag

- [ ] Sign `dist/SHA256SUMS` with the release key:

      ```bash
      gpg --armor --detach-sign dist/SHA256SUMS
      ```

- [ ] Commit the checklist result and tag the reproducible commit:

      ```bash
      git tag -s v0.1.0 -m "LMPU v0.1.0"
      git push origin v0.1.0
      ```

- [ ] Attach `dist/*.whl`, `dist/*.tar.gz`, `SHA256SUMS`, its signature, and
      `sbom.json` to the release.

## 8. Post-release

- [ ] Verify `uv pip install lmpu==0.1.0` from a clean environment.
- [ ] Archive the benchmark artifact directory used for the release report.
- [ ] Record accepted limitations (C1 model-budget partial; deferred Gate E).
