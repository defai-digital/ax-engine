# Security Policy

## Reporting a Vulnerability

Please report suspected security vulnerabilities privately — do not open a
public GitHub issue for a security report.

- Preferred: use [GitHub Security Advisories](https://github.com/defai-digital/ax-engine/security/advisories/new)
  for this repository.
- Alternative: email <enquiry@defai.digital> with a description of the issue,
  the affected version or commit, and reproduction steps if available.

We aim to acknowledge new reports within 5 business days and to keep you
updated as we investigate. Please give us reasonable time to investigate and
ship a fix before any public disclosure.

## Supported Versions

AX Engine ships from a single, fast-moving line of development (see
`CHANGELOG.md`). Security fixes are made against the latest release; we do
not maintain parallel long-term-support branches. If you are running an
older version, please upgrade before reporting an issue that may already be
fixed.

## Scope

AX Engine is a local- and LAN-first LLM inference runtime for Apple Silicon.
Two things are worth calling out that are distinct from classic
memory-safety CVEs:

- **Model weights and prompts are untrusted input.** Loading arbitrary
  downloaded model weights, or serving arbitrary prompts, carries its own
  trust-boundary considerations (e.g. resource exhaustion from adversarial
  inputs) beyond conventional code-execution vulnerabilities. Please note
  this distinction in your report.
- **Server hardening posture.** `docs/SERVER.md` is the authoritative
  statement of what the HTTP/gRPC server currently does and does not harden
  by default (e.g. authentication and rate limiting are opt-in). Please
  review it before filing a report about default configuration — the current
  scope is documented there, not hidden.

`crates/ax-engine-mlx` and `mlx-sys` are the only parts of the workspace that
use `unsafe` Rust (the documented MLX C FFI boundary); reports touching
memory safety there are especially welcome.
