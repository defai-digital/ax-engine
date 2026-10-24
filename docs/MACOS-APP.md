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
honors an explicit executable in Settings or `AX_ENGINE_BIN`, then searches
PATH and standard install directories, the complete AX Code managed installation
record, an app-bundled runtime, and a source-build runtime hint. It also discovers
user Python and virtual-environment installs. AX Code's install marker is read
from `$XDG_STATE_HOME/ax-code/ax-engine/install.json` (default
`~/.local/state/ax-code/ax-engine/install.json`); unfinished installations are
not selected. The app does not update or modify AX Code's installation.
Settings shows the resolved executable path and can override it explicitly.
Local source builds record an existing checkout native runtime (or `.venv`
console entry point) as a fallback; that development hint is not a bundled runtime. It does not install packages or download models on first launch.

An optional `--runtime-dir <directory>` bundles an already prepared standalone
runtime payload. Its executable/dylib layout must match
`scripts/prepare-standalone-release.sh`. Model download also requires the AX Engine
Python download helper and its dependencies; copying native binaries alone does
not create a self-contained downloader. Installed Homebrew runtimes provide the
complete supported command environment.

## Workflow

1. **Models:** Local Library shows existing managed, imported and cached models
   by default. As in the TUI, one snapshot per cached repository is shown,
   preferring complete weights before the newest revision. Metadata-only caches
   and missing indexed shards are marked incomplete. Add an existing folder
   without moving it, or use Catalog to browse the installed runtime's profiles.
   Revisions are retained in download requests. Catalog inclusion is not a
   memory-fit or qualification claim.
2. **Models → Transfers:** choosing Download in Catalog switches to Transfers,
   where you can view progress, cancel and retry. This is the current session's
   queue; an empty queue does not mean no models are downloaded. Transfers run
   sequentially through `ax-engine download --progress-json`. Ready requires
   complete weight files, a successful terminal record and a successful process
   exit, not just a 100% progress bar. There is no separate Downloads sidebar page.
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

## Settings navigation

Settings is a page in the main workspace. Use its sidebar row, the AX Engine
menu or Command-Comma; all three routes open the same page without a separate
settings window. The command also reopens the main window if it was closed.
Draft edits survive page switches. Save validates and persists them; Revert
restores the saved values. Runtime and model operations lock settings while busy.
Third-party notices remain available from Settings and the app menu.

## Local data

Settings and the current conversation live in
`~/Library/Application Support/AX Engine/`. Downloaded model folders default to
its `Models/` subdirectory. The library also discovers Hugging Face snapshots and
explicitly imported folders. Complete weight files are not proof that a model will load; the runtime performs admission and compatibility validation.

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
Original license and notices ship inside the app and are accessible through
**Third-Party Notices** in Settings or the AX Engine menu. Product screens,
icons and the About panel use AX Engine branding; upstream acknowledgements
are collected in that dedicated notices window.
AX Engine provides inference; MTPLX inference code and model assets are not bundled.

## Dashboard and discovery

Dashboard replaces the static Overview. It keeps one layout and changes its
primary action for discovery, downloaded files, startup, a ready service, and
failures. It shows the current model, service ownership, last successful health
check, startup elapsed time and a small snapshot of existing runtime metrics.
When disconnected, the snapshot explains how to start or connect instead of
showing an empty metric grid. Unknown metrics show a dash. Startup phases,
percentages and generated-token throughput are not invented. Server retains
lifecycle configuration, advanced counters and output.

Optional **This Mac monitoring** shows five minutes of CPU, GPU and unified-memory
history. It is off by default; the preference persists locally. Sampling runs only
while Dashboard or Server is visible in the active app, with a five-second pause
between samples and a two-second deadline per probe. CPU uses process averages
normalized by logical cores; memory uses active, wired and compressed pages.
GPU utilization is available only when IOAccelerator exposes it. These host-wide
estimates are separate from runtime metrics and do not establish model fit or
inference performance. OS memory pressure remains unknown until a public system
event arrives. Missing, stale and interrupted samples leave gaps instead of zeros.

Server's collapsed **Runtime history** uses existing service polls for MLX active
memory, active streams and interval MTP draft acceptance. Idle intervals, counter
resets and probe failures leave gaps; acceptance is not a speedup measurement.
History resets when the connection or reported model changes. All measurement
history is bounded and kept in memory only; no telemetry is sent.

Library discovery runs in the bundled `AXEngineModelScanner` helper, which only
reads metadata. Results arrive incrementally, ordinary local repositories are
visited before linked repositories, and a scan has a 20-second deadline plus
bounded process cleanup. Stop scan cancels the helper, not the inference runtime.
A new scan reaps the old worker before starting another. Existing rows remain
visible during refresh; unchecked rows are labeled and cannot be selected until
checked. Partial scans retain results and offer Retry instead of claiming an
empty library. A complete scan removes entries no longer found. Slow/unavailable
network storage may require reconnecting the volume before retrying.

Model rows show a compact name, source and download status, with full paths and
revision details behind a disclosure. Complete files do not establish runtime
compatibility, and absence from the runtime catalog does not mean unsupported.
Library filters distinguish all folders, catalog matches and other folders without
claiming compatibility. Search offers a clear recovery action when nothing matches.
The app-owned running model is labeled explicitly. Dashboard prioritizes active
transfers over older failures, and Catalog links an active download to Transfers.
The app performs compatibility validation when the operator explicitly starts a
model. Models from other tools are not silently deleted or hidden.

Use Command-1 through Command-5 to navigate Dashboard, Models, Server, Chat and
Settings; Command-F focuses model search and Command-R refreshes models.
Command-Comma still opens inline Settings. Search, selection and chat drafts
survive page switches and main-window reopening for the current app session.
API credentials remain in memory, and starting/connecting never happens merely
because a page was opened. Attached services retain Disconnect-only ownership.
