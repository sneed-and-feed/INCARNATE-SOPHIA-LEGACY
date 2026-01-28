# Release Note — Quantum Sovereignty v4.3 (One Page)

**Title:** Quantum Sovereignty v4.3 — The Pleroma Stack (Public Release)  
**Tagline:** An auditable, poetic grimoire of Vector Symbolic Architecture, Trotterized annealing, and dozenal obfuscation.

## Summary
This release packages the **Pleroma Stack** as a single signed artifact with embedded manifesto, reproducible build metadata, and a machine‑readable safety audit. The project is intentionally semantically opaque (dozenal obfuscation) — this is obfuscation, not cryptographic protection. Use with caution.

## Highlights
- **Core**: VSA binding ops, 12D tensor logic, Trotterized annealer.
- **Interoperability**: Conditional export gates for torch/numpy via `gateways.py`.
- **Provenance**: Signed `manifest.json` with commit, tag, build_timestamp, and sbom_hash.
- **Safety**: `sovereign_cli.py --safety-audit --format json` emits machine‑readable isolation and telemetry checks.

## Release Artifacts
- Binary (signed)
- `manifest.json`
- `manifest.json.sig`
- SBOM
- `tests/test-results.xml`
- `SECURITY.md`
- Appendix A (Auditors)

## Provenance Snippet (Example)
```bash
commit=874b5ea; tag=v4.3; build_timestamp=2026-01-28T13:30:00Z; sbom=sha256:abcdef...
```

## Quick Verification (Auditor)
1.  **Verify signature**: `gpg --verify manifest.json.sig manifest.json` using published PGP key.
2.  **Confirm commit**: compare `manifest.json.commit` to GitHub commit.
3.  **Run safety audit**: `python sovereign_cli.py --safety-audit --format json` and compare to manifest.

## Tone
Poetic presentation preserved; every mythic claim maps to concrete modules (see Appendix A).
