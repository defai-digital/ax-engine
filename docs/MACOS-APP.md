# AX Engine for macOS

The native SwiftUI app provides local model browsing, downloads, server controls,
streaming chat and runtime metrics. The CLI and `ax-engine tui` remain available.
The app targets Apple Silicon and macOS 26 or later. Inference stays in the
AX Engine runtime; the GUI does not change model support or certification.

## Build and open

Install Xcode with the macOS 26 SDK and Swift 6 support, plus Python 3.12 or
later with Pillow 13.0.0. From this checkout:

```bash
swift test --package-path packaging/macos/AXEngineApp
scripts/build-macos-app.sh
open "build/macos/AX Engine.app"
```

Use `--release` for an optimized app build. The build is locally ad-hoc signed;
it is not a published or notarized distribution. Developer ID signing,
notarization and signed updates remain release work. The icon is generated from
the supplied original artwork, with the same composition-preserving safe-zone,
rounded-plate and multi-resolution checks used by AX Code Desktop.

## Runtime setup

Install AX Engine using the main README's installation instructions. The app
looks for a bundled runtime, `/opt/homebrew/bin/ax-engine`, or
`/usr/local/bin/ax-engine`. Settings can select another executable by absolute
path. It does not install packages or download models on first launch.

An optional `--runtime-dir <directory>` bundles an already prepared standalone
runtime payload. Its executable/dylib layout must match
`scripts/prepare-standalone-release.sh`. Model download also requires the AX Engine
Python download helper and its dependencies; copying native binaries alone does
not create a self-contained downloader. Installed Homebrew runtimes provide the
complete supported command environment.

## Workflow

1. **Models:** refresh the installed runtime's catalog. Revisions are read from
   the runtime and retained in download requests. Add a local folder, or choose
   Download. Catalog inclusion is not a memory-fit or qualification claim.
2. **Downloads:** view progress, cancel and retry. Transfers run sequentially
   through `ax-engine download --progress-json`. Ready requires both a successful
   terminal record and a successful process exit, not just a 100% progress bar.
3. **Server:** choose an installed folder or cached alias. Start uses offline
   resolution and loopback only. Runtime MTP and expert-streaming defaults remain
   unchanged. An app-owned server uses a per-launch in-memory API key.
4. **Chat:** select a running model, send text, inspect reasoning when supplied,
   copy replies, retry or stop. Command-Return sends; Return enters a new line.
   Streaming text is coalesced; settled replies receive basic Markdown/code-block
   rendering. Generated HTML and tool calls are never executed.
5. **Server metrics:** inspect live runtime counters. Missing measurements appear
   as unavailable. Counters are not a benchmark or a throughput certification.

Connect attaches to an existing AX Engine service on the configured local port.
Its API key stays in memory. Disconnect and app quit do not stop an attached
service. Quit asks before stopping app-owned inference or downloads. Closing a
window follows macOS app conventions; use Quit to stop app-owned work.

## Local data

Settings and the current conversation live in
`~/Library/Application Support/AX Engine/`. Downloaded model folders default to
its `Models/` subdirectory. The library also discovers Hugging Face snapshots and
explicitly imported folders. A manifest-present label is not proof that a model
will load; the runtime performs admission and compatibility validation.

Only app-managed model directories can be deleted from the GUI, after
confirmation and while disconnected. Imported directories and the Hugging Face
cache are not deleted. Chat text stays on disk locally; Clear removes the saved
conversation. Diagnostic output is bounded in memory. The app sends no telemetry.

## Scope and credits

This version provides text chat and local inference management. Model conversion,
fan control, web-search tools and coding-agent execution remain outside this app.
Runtime installation and updates use AX Engine's existing distribution channels.

Powered by MTPLX: <https://github.com/youssofal/mtplx>.
Selected native app components are adapted from MTPLX by Youssof Altoukhi under
Apache-2.0, at revision `9882703f3105363ddc37eca9f97aa09a1d387112`.
Original license and notices ship inside the app and are accessible in Settings.
AX Engine provides inference; MTPLX inference code and model assets are not bundled.
