# Minisign release signing

AX Engine publishes detached minisign signatures for GitHub Release and Homebrew
assets. The release scripts (`scripts/publish-github-release.sh`,
`scripts/brew-release.sh`) sign artifacts before upload via
`scripts/minisign-artifact.sh`; the shared AX product signing key lives outside the
repository in `~/signkey` by default.

The secret key must never be committed, logged, or uploaded as a release asset.
The public key is safe to publish so users can verify downloaded artifacts.

## Key files

Default paths (the shared AX product signing key):

```text
~/signkey/ax.sec
~/signkey/ax.pub
```

Override them with `AX_MINISIGN_SECRET_KEY` / `AX_MINISIGN_PUBLIC_KEY` or the
`--secret-key` / `--public-key` flags.

## Generate the keypair

```bash
bash scripts/minisign-keygen.sh
```

`minisign` prompts for a password and writes the keypair to `~/signkey`. The
script sets the directory to mode `700`, the secret key to `600`, and the public
key to `644`. It refuses to overwrite an existing keypair unless you pass
`--force` (key rotation).

For a short-lived local test key only, generate an unencrypted key:

```bash
bash scripts/minisign-keygen.sh --allow-unencrypted-test-key
```

Do **not** use an unencrypted key for releases. Use `--dry-run` to preview.

## Pin the release public key (recommended)

`scripts/minisign-artifact.sh` can refuse to sign or verify unless the local
public key matches an expected value, so a rotated, wrong, or planted key cannot
silently produce valid-looking signatures. The pin is env-sourced because the
shared AX public key material is intentionally not stored in this repo.

Set the pin in your shell rc or CI secret:

```bash
export AX_MINISIGN_PINNED_PUBLIC_KEY='RWS...your-shared-ax-public-key...'
```

When set, the signer reads the public key material from `~/signkey/ax.pub`
and fails closed on mismatch. When unset, pinning is not enforced (current
behavior), which keeps unconfigured local recovery flows working.

To override per-invocation:

```bash
scripts/minisign-artifact.sh --pinned-public-key 'RWS...' artifact.tar.gz
```

## Sign release artifacts

The release scripts sign by default. To sign manually:

```bash
scripts/minisign-artifact.sh path/to/artifact.tar.gz
```

It creates `artifact.tar.gz.minisig` beside the artifact and verifies it against
the public key. Useful flags:

```text
--secret-key <path>         Override the secret key path.
--public-key <path>         Override the public key path.
--public-key-string <key>   Verify with a raw public key string (no file needed).
--signature-dir <dir>       Write all .minisig files into this directory.
--trusted-comment <text>    Override the trusted comment.
--untrusted-comment <text>  Override the untrusted comment.
--force                     Overwrite an existing .minisig.
--no-verify                 Skip post-sign verification.
--dry-run                   Print the plan; sign nothing.
--keychain-service <svc>    Keychain service name (default: ax-minisign).
--keychain-account <acct>   Keychain account name (default: ax-release).
```

By default each trusted comment includes the artifact basename, SHA-256 digest,
and a UTC signing timestamp:

```text
ax-engine artifact ax-engine-v6.4.5-macos-arm64.tar.gz sha256=<digest> signed=2026-06-16T12:00:00Z
```

## Verify an artifact

Users verify a downloaded artifact against the published public key:

```bash
minisign -V \
  -P 'RWS...published AX public key...' \
  -m ax-engine-v6.4.5-macos-arm64.tar.gz \
  -x ax-engine-v6.4.5-macos-arm64.tar.gz.minisig
```

## macOS Keychain for local signing

For local release recovery on macOS, keep the encrypted minisign secret key on
disk with mode `600` and store only its passphrase in Apple Keychain:

```bash
security add-generic-password -U \
  -a ax-release \
  -s ax-minisign \
  -w
```

`scripts/minisign-artifact.sh` reads that Keychain item automatically when
`AX_MINISIGN_PASSWORD` is not set. The password resolution order is:

```text
AX_MINISIGN_PASSWORD  >  macOS Keychain  >  interactive minisign prompt
```

Override the lookup names with `AX_MINISIGN_KEYCHAIN_SERVICE` /
`AX_MINISIGN_KEYCHAIN_ACCOUNT` or the matching flags.

## CI secrets

When signing in CI, provide the key and passphrase as repository secrets (names
match the `AX_MINISIGN_*` env contract consumed by the release scripts):

```text
AX_MINISIGN_SECRET_KEY          path or contents of the minisign secret key
AX_MINISIGN_PASSWORD            passphrase that unlocks the key
AX_MINISIGN_PINNED_PUBLIC_KEY   expected public key (fail-closed pin)
```

## Scope

This minisign signature is separate from platform code signing and notarization.
It proves artifact integrity against the published minisign public key; it does
not replace Apple Developer ID signing/notarization. See `scripts/README.md` for
the `publish-github-release.sh --sign-identity` release path.
