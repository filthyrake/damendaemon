# Contributing to damendaemon

Thanks for your interest. This document covers how to report problems, set up
a development environment, run the tests, and land a pull request.

## Reporting bugs and requesting features

- Search [existing issues](https://github.com/filthyrake/damendaemon/issues)
  before opening a new one.
- Bug reports should include a minimal reproduction, the expected and actual
  behavior, and your environment (OS, Python version, GPU if relevant).
- Feature requests should state the capability, why existing mechanisms do not
  cover it, and any measurement or evidence that motivates it.
- Performance claims must come from real runs on real hardware with identical
  model revision, precision, context, and output lengths; report latency and
  throughput separately. Mocked or CPU-only runs do not establish GPU
  performance.

## Security

Do not open a public issue for security problems. Use
[GitHub private vulnerability reporting](https://github.com/filthyrake/damendaemon/security/advisories/new)
instead. See [SECURITY.md](SECURITY.md) for scope.

## Development setup

Python 3.11 or 3.12. The core package has no required dependencies.

```bash
git clone https://github.com/filthyrake/damendaemon.git
cd damendaemon
python -m pip install -e .
```

Optional extras are declared in `pyproject.toml`:

```bash
python -m pip install -e ".[runtime]"      # inference runtime
python -m pip install -e ".[validation]"   # tokenization contract checks
python -m pip install -e ".[gguf]"         # GGUF/sentencepiece support
```

Some checks use CUDA or ROCm GPUs; they are opt-in experiments under
`scripts/experiments/` and are never part of the CI gate.

## Running the tests

The CI gate is CPU-only. Locally:

```bash
# Dependency-free contracts (matches the CI contracts job)
python -S -m unittest \
  tests.test_qualification_optimization tests.test_verifier_http_errors \
  tests.test_report_io tests.test_error_state tests.test_token_workload \
  tests.test_manifest_preflight tests.test_checkpoint_json \
  tests.test_tokenizer_metadata tests.test_stopping tests.test_quota_timeout \
  tests.test_bench

# Runtime tests (requires the runtime extra)
python -m unittest tests.test_runtime tests.test_sampling tests.test_server
```

Run the modules you touched plus the contracts job before opening a PR. The
full CI suite also runs the contracts job under `-O` and a broader integration
list (see `.github/workflows/cpu-contracts.yml`).

## Pull requests

- Pull requests are required for all changes to `main`; continuous-integration
  checks must pass before merge:
  - `Python 3.11 contracts` and `Python 3.12 contracts`
  - `CPU runtime and tokenizer integration`
  - `Gitleaks secret scan`
- Keep changes focused: no drive-by refactors or reformatting, and add the
  tests your change requires.
- Commit messages use conventional prefixes (`feat:`, `fix:`, `docs:`,
  `chore:`, `test:`, `refactor:`) with an imperative summary.
- Do not commit credentials, model weights, or large binary artifacts. Model
  weights stay outside the repository; evidence archives under `docs/evidence/`
  are measurement JSON only.

## License

By contributing, you agree that your contributions are licensed under the
repository's [license](LICENSE.md).
