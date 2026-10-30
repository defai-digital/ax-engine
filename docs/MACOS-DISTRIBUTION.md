# Distributing AX Engine for macOS

Status: signed GUI release tooling; public availability is verified separately
from runtime publication using `verify-public` below.
Scope: macOS 26.2 or later, Apple Silicon arm64.
Last reviewed: 2026-10-26.
Owner: AX Engine maintainers.
Related: [native app](MACOS-APP.md), [installation guide](GETTING-STARTED.md).

The native GUI uses the curl bootstrap as its primary end-user installation
entry point, installing a self-contained signed DMG payload. Users do not need Homebrew,
Python, a Hugging Face account or a Cloudflare account. The existing CLI
installer and Homebrew formula remain separate products. A generated optional
`ax-engine-gui` Cask installs the exact same DMG.

## Local build and Apple qualification

Run from a clean checkout with the pinned Rust toolchain, Swift/macOS SDK,
pinned pip MLX runtime, Python 3.12 or later and Pillow installed. Python and
Node are build/release tools only; they are not shipped in the app.

```bash
python3 scripts/release-macos-app.py build \
  --output build/gui-candidate \
  --mlx-lib-dir .venv/lib/python3.12/site-packages/mlx/lib
python3 scripts/release-macos-app.py package \
  --source build/gui-candidate --output build/gui-notarized \
  --notary-profile ax-notary
python3 scripts/release-macos-app.py verify --source build/gui-notarized
```

Use fresh output directories. `build.json` binds the source commit, version,
Rust/MLX pins and complete app contents. Packaging rejects a changed app.
The app contains its runtime in `Contents/Helpers/AXEngineRuntime.bundle`,
with native code in `MacOS` and Metal data/license in `Resources`. Its internal
Metal symlink preserves MLX's normal resource lookup. It does not contain
checkout paths, models, tokens or a Python environment.

Packaging signs each native image, the runtime bundle and the outer app in that
order with Developer ID, hardened runtime and secure timestamps. All code uses
Apple Team `N5ZUZDUJS6`; no library-validation bypass or JIT entitlement is added.
The app and DMG are submitted separately to Apple, stapled and checked with
Gatekeeper. Notary request IDs/status records are retained if a wait fails;
inspect them with `notarytool info` or `log` before retrying in a fresh directory.
No notarization result is inferred from a successful local codesign operation.

`verify` mounts the DMG read-only and checks the actual app, source identity,
contents, notarization, signatures and bundled native-download capability.
Each release must additionally qualify Finder launch and a quarantined
fresh install/upgrade on a supported Mac. Synthetic tests and local signing do
not establish model performance, SKU qualification or a public release.

## Signed release inventory

```bash
node packaging/macos/distribution/distribution.mjs prepare \
  build/gui-notarized build/gui-downloads
node packaging/macos/distribution/distribution.mjs verify build/gui-downloads
```

The release key is pinned to `docs/ax-minisign.pub`. The existing Minisign helper
reads the private key and a passphrase supplied through the environment or
Keychain; secrets are never command-line arguments or release assets.
Preparation signs the DMG, Apple receipt, installer, Cask, inventory, manifest
and readiness record. Every manifest binds the same source commit/version.
`latest.json` contains a readiness payload and its signature together, so an
atomic R2 replacement cannot mix two versions' pointer/signature files.

The end-user command is:

```bash
curl -fsSL https://download.automatosx.com/ax-engine-desktop/install | bash
```

The complete bootstrap is parsed before execution, so a truncated curl stream
cannot begin installation. `--help` works without network access. Installation
prints verification, download and destination stages, then explains how to open
the app. Users rerun the command to upgrade after quitting the app; it does not
add global CLI commands to PATH. Optional Cask generation is retained for
release compatibility, not required by the end-user installation flow.

The installer checks its pinned Minisign verifier, signed pointer/readiness,
manifest, package size/hash, Apple Team, bundle ID, source commit and Gatekeeper
before changing the application. `--verify-only` performs the download and
verification without installation. `--applications-dir /absolute/directory`
allows an existing user-writable Applications directory; otherwise `/Applications`
is used with elevation only when required. No Xcode installation is requested.

Installation uses a lock, staged copy and previous-app backup. It refuses a
version downgrade, does not overwrite another app or follow a linked destination,
and restores the previous app on a failed swap or installed-app verification.
A subsequent run can recover an interrupted swap. Stale-lock recovery is
serialized, and processes owned by another user are treated as active. If a
process is killed while holding `.AX-Engine.recovery.lock`, inspect active
installers before manually removing that empty guard directory. Quit AX Engine first. Models,
settings and conversation files are outside the replacement and remain untouched.
The installer never removes quarantine or disables Gatekeeper.

## Isolated Cloudflare channel

The GUI channel base is
`https://download.automatosx.com/ax-engine-desktop`. A version is available only
after its signed metadata and public readback pass the publication steps below.
It uses its own `ax-engine-desktop-releases` bucket and
`ax-engine-desktop-downloads` Worker. It does not change AX Code/AX Code Desktop
buckets, routes, installers, release objects or latest pointers.

Operator credentials come from `CLOUDFLARE_API_TOKEN` or the user-selected local
`~/.cf` token file. No credential is copied into the Worker configuration or app.
`gateway.mjs inspect` is read-only. The other commands below perform explicit
remote writes and are not part of a local build/test run.

```bash
node packaging/macos/distribution/gateway.mjs inspect
node packaging/macos/distribution/gateway.mjs provision
# After the stable Engine release tag and exact-source CI are public:
node packaging/macos/distribution/gateway.mjs stage build/gui-downloads CI_RUN_ID
node packaging/macos/distribution/gateway.mjs enable build/gui-downloads CI_RUN_ID
node packaging/macos/distribution/distribution.mjs publish build/gui-downloads CI_RUN_ID
node packaging/macos/distribution/distribution.mjs verify-public build/gui-downloads
```

Provisioning deploys a disabled, path-scoped gateway with no workers.dev/preview
alias. It rejects another Worker's ownership of the route. Stage uploads only
immutable version objects, checks collisions and reads back every byte. Enable
requires the complete signed version to be staged; it does not update latest.
Publication requires the matching local/remote stable release tag, a published
GitHub release and successful exact-source `CI` run. A local candidate whose
commit differs from the published version cannot overwrite that version.

Use one publisher at a time across all machines. The local publication lock is
not a distributed lock. The publisher checks immutable collisions before writes,
performs authenticated and anonymous curl readback, refuses downgrades or changed
published versions, and writes latest last. On a failed pointer readback it
restores the old pointer only if the remote pointer still belongs to this run.
After an uncertain network failure, inspect remote state and resume the same
signed directory; never regenerate signatures into a partially published version.
For the first release there is no previous pointer to restore; all immutable
bytes are verified before the first latest object can be written.

The REST uploader fails closed for files above 300 MiB. A larger artifact needs
a separately qualified multipart transport; do not split the DMG or bypass the
size check. The Worker supports GET/HEAD, conditional reads and byte ranges, binds
response data to the metadata ETag, uses immutable caching for version objects,
and never caches the mutable root pointer or installer.

The local canonical deployment wrapper exposes these tools through
`.internal/deploy/deploy --gui`: `build`, `package`, `verify`,
`prepare-downloads`, `verify-downloads`, `gateway`, `publish`, and `verify-public`.
CLI/PyPI/formula publication remains unchanged. Native Sparkle updates use the
same release channel; see [macOS app updates](MACOS-UPDATES.md) for first-release
activation, signing and validation. Manual DMG/curl/Cask upgrades remain supported.

## Validation

```bash
python3 -m unittest scripts/test_release_macos_app.py scripts/test_macos_gui_install.py
node --test packaging/macos/distribution/distribution.test.mjs
```

Tests exercise real Minisign signatures using temporary test keys, tampered
metadata, immutable collisions, partial publication, stale pointers, rollback,
installer interruption/recovery, downgrade rejection and gateway range/ETag
behavior. Installer filesystem tests use deterministic platform/network fixtures;
they do not substitute for real Gatekeeper, notarization or live R2 verification.

The packaging sequence follows Apple's
[code-signing guidance](https://developer.apple.com/library/archive/technotes/tn2206/_index.html)
and [notarization workflow](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution).
Gateway reads use the [R2 Workers API conditional operations](https://developers.cloudflare.com/r2/api/workers/workers-api-reference/).
