# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Use GitHub's
**Report a vulnerability** button (Security tab → Report a vulnerability) to
submit a private report. Include reproduction steps and affected paths if
possible.

## Scope

- The runtime, serving, and CLI code in this repository.
- The CUDA/ROCm kernels under `native/` and `third_party/` adapters.
- The GitHub Actions workflows under `.github/`.

Out of scope: model weights (not stored here), third-party model licenses in
`docs/licenses/`, and benchmark evidence archives.

## Test credentials

`tests/fixtures/host_tls/` contains no live credentials. TLS tests generate
fresh short-lived keys at runtime (see the README in that directory); any keys
that appear in Git history are synthetic, test-only, and never trusted by any
deployed service.
