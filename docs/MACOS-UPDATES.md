# macOS app updates

AX Engine includes a native updater in builds that ship Sparkle. Use **AX Engine →
Check for Updates…**, or **Settings → Updates**. Automatic checks are opt-in. A
check contacts the AX Engine download service; it sends no conversations, model
files or usage analytics. Downloads and installation require user action.

The updater verifies the archive signature before replacing the application.
Restarting uses the app's normal quit flow: active app-owned inference, model
operations and unsaved settings prompt for confirmation. Cancel keeps the app
running. Accepted restarts wait for cleanup; attached services keep running.
Models, settings and conversations are outside the application bundle and are
preserved. Development builds cannot update themselves.

The already published v8.0.0 and earlier apps do not contain this updater. Install
the first updater-enabled release with the existing curl installer or DMG; later
releases can be installed from the app. The source change alone does not publish
an update feed or upgrade an installed application.

## Release integration

The Swift package pins Sparkle and commits its resolution. The bundle builder
embeds the framework, retains its license, thins its native code to arm64 and
signs nested code before enclosing bundles. Release packaging applies Developer
ID signatures and notarization as before. App metadata pins the update feed and
public Ed25519 key from `packaging/macos/sparkle.json`.

The dedicated Sparkle private key stays in the release operator's Keychain under
the account named in that configuration. It is separate from the existing
Minisign identity. Resolve the Swift package to obtain the pinned Sparkle tools.
`generate_keys --account ax-engine-desktop -p` must print the pinned public key;
never generate a replacement key to bypass a signing failure. Back up or transfer
this identity through the operator's secure credential process. Private keys
must never enter the repository or release artifacts.

`distribution.mjs prepare` signs the final notarized DMG with Sparkle's
`sign_update`, independently verifies its Ed25519 signature, and generates a
canonical `appcast.xml`. Inventory v2 binds that feed alongside the existing
Minisign-signed release assets. Historical v1 inventories remain verifiable.

Before publishing the first updater-enabled release, deploy the download gateway
with the `appcast.xml` route using the existing GUI gateway workflow. The route
resolves the current version from the atomic `latest.json` pointer and redirects
to its immutable feed. A missing feed fails closed. All assets must pass anonymous
readback before cutover; failure of the updater alias after cutover restores the
previous pointer. Rollback prevents new discovery; it does not revoke an archive
already downloaded by a client. Never replace assets for a published version.

The canonical GUI `publish` and `verify-public` operations verify the updater
alias as well as the immutable feed and DMG. The initial release must pass these
gates before its updater-enabled app is offered to users. Before that publication,
a local release-style candidate checking the public URL will receive a feed error;
use the loopback integration fixture below for local installation tests.

## Validation

```bash
swift test --package-path packaging/macos/AXEngineApp --no-parallel
node --test packaging/macos/distribution/distribution.test.mjs
python3 -m unittest scripts/test_release_macos_app.py scripts/test_macos_gui_install.py
AX_GUI_UPDATER_INTEGRATION=1 python3 -m unittest scripts/test_macos_updater_integration.py
```

The opt-in integration suite compiles a headless test client, serves a temporary
feed on loopback, and updates disposable bundles with ephemeral signing keys.
It checks an actual download/install/relaunch, delayed graceful shutdown,
cancellation, an already-current version and a rejected archive signature. It
never replaces `/Applications/AX Engine.app` or uses production signing keys.
Production Developer ID, notarization and public-channel upgrade qualification
remain release gates.
