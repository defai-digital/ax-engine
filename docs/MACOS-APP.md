# AX Engine for macOS

The native SwiftUI app provides local model browsing, downloads, server controls,
streaming chat and runtime metrics. The CLI remains available; the TUI is retired.
The app targets Apple Silicon and macOS 26.2 or later. Inference stays in the
AX Engine runtime; the GUI does not change model support or certification.

## Install and update

Install the signed app and its bundled runtime with:

```bash
curl -fsSL https://download.automatosx.com/ax-engine-desktop/install | bash
```

Quit AX Engine first. Open AX Engine from Applications after installation.
Homebrew, Python and Xcode are not required. Run the same command again to
upgrade; models, settings and conversations are preserved. Use
`bash -s -- --verify-only` to verify the download without installing. Automatic
app updates remain future work. See the [README](../README.md#install-ax-engine-for-macos)
for requirements and installer options.

## Build and open from source

Install Xcode with the macOS 26 SDK and Swift 6 support, plus Python 3.12 or
later with Pillow 13.0.0. From this checkout:

```bash
swift test --package-path packaging/macos/AXEngineApp
scripts/build-macos-app.sh
open "build/macos/AX Engine.app"
```

Use `--release` for an optimized app build. The build is locally ad-hoc signed;
it is not a published or notarized distribution. Developer ID signing,
notarization and public cutover use the separate [distribution tooling](MACOS-DISTRIBUTION.md).
A local source build does not activate that channel. Automatic signed updates
remain future work. The icon is generated from
the supplied original artwork, with the same composition-preserving safe-zone,
rounded-plate and multi-resolution checks used by AX Code Desktop.

## Runtime setup

The curl distribution bundles its runtime. For a source build, install a runtime
using the README's advanced CLI/Python instructions or bundle one explicitly. The app
honors an explicit executable in Settings or `AX_ENGINE_BIN`, then searches
an app-bundled runtime, PATH and standard install directories (including the curl
installer at `~/.local/ax-engine/ax-engine`), the complete AX Code managed
installation record, and a source-build runtime hint. It also discovers
user Python and virtual-environment installs. AX Code's install marker is read
from `$XDG_STATE_HOME/ax-code/ax-engine/install.json` (default
`~/.local/state/ax-code/ax-engine/install.json`); unfinished installations are
not selected. The app does not update or modify AX Code's installation.
Settings shows the resolved executable path and can override it explicitly.
Local source builds record an existing checkout native runtime (or `.venv`
console entry point) as a fallback; that development hint is not a bundled runtime. It does not install packages or download models on first launch.

An optional `--runtime-dir <directory>` bundles an already prepared standalone
runtime payload. Its executable/dylib layout must match
`scripts/prepare-standalone-release.sh`. Source builds now download public model
packs through the native Rust runtime, without Python or a Hugging Face account.
Use a runtime built from the same source revision as the GUI. Older published
runtimes may still require their Python download helper. `--distribution` requires
a staged runtime and omits checkout-specific development paths from the app.
Python and Pillow are build-time icon tools, not end-user download dependencies.

## Workflow

**First run.** When a complete library scan finds no models, a short guide
opens: what AX Engine does, a list of catalog models (download size within
this Mac's memory first, smallest first; a size comparison, not a fit
claim), the download with live progress, and **Start Server**. Closing it
keeps any download running. It is offered once; reopen it from **Help ›
Getting Started**.

1. **Start serving (Dashboard):** the **Inference service** card at the top of
   Dashboard is the quickest way to serve: pick a downloaded model and press
   **Start Server** (or Shift-Command-R, or the menu-bar item). With nothing
   downloaded yet, the card says so and offers **Download a Model**, which
   opens the Catalog. The model picker is searchable and groups models under
   **Favorites** (star a row to pin it), **Recent** (the last five models you
   started) and the rest; arrow keys and Return work from the search field,
   and **Download more models** opens the Catalog. Model names are short
   titles (for example **Tiel Coder 35B A3B**) with quantization and MTP as
   tags; the full name and path are in the tooltip. With nothing chosen, the
   running model, then the most recently started one, a favorite or the first
   download is preselected. Before anything is listening the card says **Not
   listening**; while starting it shows the elapsed time and the runtime's
   latest output line, and **Cancel Start** stops the launch. Once running it
   offers **Restart Server**, **Stop Server** (Shift-Command-Period), **Copy
   API URL**, **Copy API Key** when the server uses one, and **Open
   Playground**. **Serving options** folds the serving mode and request cap
   behind a one-line summary such as "Automatic mode · up to 4 requests · no
   API key". By default the server needs no API key: it listens on 127.0.0.1
   only, so local clients can call it directly.
2. **Server Settings:** the same status, model picker and lifecycle buttons,
   with the serving options and API key always visible. **Require an API key**
   gives each later launch a fresh in-memory key, copied with **Copy API
   Key**; while a server is running the toggle shows **Restart required** and
   **Restart Now**. **Serving mode** (Automatic, Coding, Agentic, Chatbot)
   selects the runtime's speculation profile for the next app-owned launch
   (`ax-engine serve <model> -- --speculation-profile <mode>`; Automatic
   passes nothing, so the runtime default and any `AX_MLX_SPECULATION_PROFILE`
   in the environment still apply). **Concurrent requests** (1–8, default 4)
   is forwarded as `--max-concurrent-requests`: one server decodes up to that
   many requests at once in a single batch and answers 429 beyond it, and the
   runtime's memory plan sizes per-request state for that count. 4 keeps each
   request near full speed on batched models; 8 is the runtime's
   batched-decode cohort limit. While running, the options show the runtime's
   batch slots in use from its own gauges. Serving mode, request cap and API
   key save as soon as they change, on Dashboard, Server Settings or Settings
   › Runtime, without touching other unsaved Settings edits, and apply at the
   next start or restart. When the running server differs from them, Restart
   names the change (**Restart with Selected Model**, **Restart in Coding
   Mode**, **Restart with New Request Cap**, **Restart to Apply API Key
   Setting**) and a note explains what keeps running until then. Attached
   services keep their own mode, limits and authentication. When Start is
   unavailable the card says why (no model downloaded, none selected, the
   selection incomplete, the server busy, or an attached service). Start uses
   offline resolution and loopback only, and refuses up front when the
   configured port is already in use instead of failing after the model has
   loaded. Runtime MTP and expert-streaming defaults remain unchanged. Stop
   and Restart ask for confirmation only while a reply is being generated.
   Stop gives the runtime eight seconds to drain before it is force-killed.
   When the runtime exits on its own, the error shows its last diagnostic line
   with **Start Again** and **Copy Diagnostics** (the bounded runtime output,
   with the API key redacted), and **Runtime output** opens. Runtime output
   can be filtered by text, narrowed to warnings and errors, and limited to
   the last 100 or 500 lines; filters change the view only, and **Copy All**
   copies the whole bounded output. Runtime metrics
   and history appear only while something is serving; Mac CPU, GPU and memory
   charts are on Dashboard. **Already running AX Engine on this Mac? Connect to
   it** (collapsed) attaches to a service on the configured port; a key you
   enter is held in memory for that connection only.
   **Copy API URL** copies the actual connected endpoint, and **Copy API Key**
   copies its current key for an external local client. The key is copied
   only on your click, stays on this device, and is cleared after 60 seconds
   if the clipboard has not changed. It is also cleared when the connection
   becomes unavailable or the window closes. Confidential and transient
   markers request exclusion from compatible clipboard histories; they do not
   prevent other local apps from reading the clipboard. The key is never
   added to saved settings. A restarted app-owned server gets a new key, so
   copy it again after restarting.
3. **Models:** three segments. **Your Models** lists app-managed and cached
   AutomatosX models; each row says **Ready to serve**, **Weights incomplete
   or metadata only** or **Not checked in this scan**, and marks the
   **Selected** and **Serving** model. One snapshot per cached repository is
   shown, preferring complete weights before the newest revision. **Serve**
   on a ready row starts the server on it when nothing is serving; while a
   server runs, **Select on Dashboard** selects it there so Restart switches
   to it. The same action is on downloaded Catalog rows and finished
   downloads. **Catalog** lists public AutomatosX packs (no Hugging Face
   account or token) grouped into those that fit this Mac's unified memory and
   a collapsed group of larger ones; each row shows the runtime's approximate
   pack size, whether MTP artifacts are included and its certification word
   when recorded, plus the free space on the model volume. These are size
   comparisons, not fit or qualification claims. A pack whose estimate exceeds
   the free space asks first: AX Engine checks the exact remaining size before
   downloading and stops without touching existing models when it does not
   fit, so **Download Anyway** is safe and a resumed download may need less.
   Download third-party models independently and use CLI local model paths.
   Arbitrary local folder imports and editable server paths are not GUI
   features. **Downloads** (the count of active ones is on the segment) shows
   each download's phase (checking files, downloading, verifying), bytes done
   of total, smoothed throughput and an estimated time left, the file being
   transferred, queue position, when it started and how long it took, and a
   warning when no data has arrived for 30 seconds, with **Restart Download**,
   which reconnects the same row and keeps verified files. Byte counts come from
   runtimes that report them; with an older runtime the sizes are catalog
   estimates and are labeled "about". Cancel asks first and keeps verified
   files for Retry or Resume; a finished row offers **Serve**, Show in Finder
   and a menu with Remove from Disk and Dismiss. Downloads run sequentially
   through `ax-engine download --progress-json`. **Downloaded** requires
   complete weight files, a successful terminal record and a successful
   process exit, not just a 100% progress bar. This is the current session's
   queue. Interrupted native downloads leave a hidden stage folder so a retry
   can resume; Downloads lists these under **Interrupted downloads** with
   their size, **Resume** when the stage matches a catalog entry, and
   **Discard**, which refuses while the runtime still holds the download lock.
   Your Models links to them.
4. **Playground:** select a running model, send text, inspect reasoning when supplied,
   copy replies or code blocks, retry or stop (Command-Period). Return or
   Command-Return sends; Shift-Return enters a new line. You can type the next
   message while a reply streams. Without a server, Playground shows one state
   with the next step: **Start Server** on the selected model, **Download a
   Model** when nothing is downloaded, or **Review Server** after an error.
   Streaming text is coalesced; settled replies receive basic Markdown/code-block
   rendering. Generated HTML and tool calls are never executed.
   History is bounded to 200 messages and 4 MiB of combined prompt, answer, and
   reasoning text. Each reply is also bounded to 2 MiB; the saved JSON remains
   at most 8 MiB, including escaped text. Reaching a limit preserves the existing
   conversation: use **Export**, then **Clear** to continue. **Export** writes
   a local JSON archive. Oversized or unreadable saved files are preserved and
   block new sends; **Reveal Saved Conversation** opens their location before
   an explicit **Clear** replaces them. These are storage/display limits, not
   a model token-context limit; the server still validates the model context.

5. **Server metrics:** inspect live runtime counters on Server Settings while a
   server runs. Missing measurements appear as unavailable. Counters are not a
   benchmark or a throughput certification.
6. **Menu bar and Server menu:** the menu-bar item (on by default; Settings ›
   Runtime › Menu bar) shows the server state and offers Start Server, Start
   with… (any downloaded model), Restart, Stop, Copy API URL, Open Playground
   and Open AX Engine. The **Server** menu has Start/Restart Server
   (Shift-Command-R), Stop Server (Shift-Command-Period), Copy API URL and
   Open Playground.

Connect attaches to an existing AX Engine service on the configured local port.
Its API key stays in memory. Disconnect and app quit do not stop an attached
service. Quit asks before stopping app-owned inference or downloads. Closing a
window follows macOS app conventions; use Quit to stop app-owned work.

## Settings navigation

Settings is a page in the main workspace. Use its sidebar row, the AX Engine
menu or Command-Comma; all three routes open the same page without a separate
settings window. The command also reopens the main window if it was closed.
Draft edits survive page switches. Save validates and persists them; Revert
restores the saved values. Runtime and model operations lock runtime locations
and the port while busy. Generation-only changes remain editable and apply to
the next Chat request without restarting the server or rescanning models.
Third-party notices remain available from Settings and the app menu.

## Playground profiles

Settings › Generation opens with **Coding**, **Chatbot**, **Media**, and **Custom**. Profiles
set a response ceiling and use runtime sampling defaults (no temperature, top-p,
top-k, or penalty overrides). These are app starting points, not model-specific
quality or performance qualifications:

| Profile | Maximum output tokens | Intended use |
| --- | ---: | --- |
| Coding | 8,192 | Code, explanations and tests |
| Chatbot | 2,048 | Everyday conversation |
| Media | 4,096 | Text responses for image/document analysis |
| Custom | User-selected | Preserve or adjust your own settings |

The response ceiling is not context size or a reserved memory allocation.
Reasoning can consume output tokens; increase the limit when necessary. The
model/runtime still enforces context and memory admission. Profiles do not change
thinking, MTP, tool execution, model loading or system prompts. Media requires a
compatible model and client; built-in Chat remains text-only. It does not enable
image, audio or video generation.

**Advanced generation** exposes the output limit and an optional temperature
override. Editing preset values displays Custom. Existing settings retain their
original temperature and output limit as Custom; fresh installs start with
Chatbot. Save commits the draft, and Revert restores it.

Profiles apply to built-in Chat requests. Other apps continue to supply their own
parameters. **Copy Request Defaults** copies the saved only a JSON request fragment
(`max_tokens` and an optional `temperature`) for configuration in a compatible
client; it contains no API key or local path and does not change a running server's
global defaults. Only supported request parameters should be used by that client.

## Local data

Settings and the current conversation live in
`~/Library/Application Support/AX Engine/`. Downloaded model folders default to
its `Models/` subdirectory. The library also discovers Hugging Face snapshots and
explicitly imported folders. Complete weight files are not proof that a model will load; the runtime performs admission and compatibility validation.

Models that AX Engine downloaded can be deleted from the GUI after a
confirmation that names the folder and the space it frees. **Remove** sits on
every removable Your Models row and in the row's context menu; the same
action is in the Downloaded menu of a Catalog row and in the menu of a
finished download.
When removal is temporarily blocked the row says why inline (the server is
serving it, a download or model operation is running, or an attached service
may be using it). Rows that AX Engine will never delete (imported folders,
`AX_ENGINE_MODELS_DIR` folders, non-AutomatosX cache entries) say so instead of
hiding the control. App-managed folders
go through `ax-engine models rm`. For an `AutomatosX/*` repository in the
Hugging Face cache the app deletes that whole `models--AutomatosX--…` entry,
including every cached revision, because the library shows one snapshot per
repository; other apps sharing that cache must download it again. Removal is
blocked for the model the app is currently serving, while attached to an
external service, and during downloads. Non-AutomatosX cache entries and
imported directories are never deleted. Each row offers Show in Finder. Chat
text stays on disk locally; Clear removes the saved conversation. Diagnostic
output is bounded in memory. The app sends no telemetry.

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

Dashboard is a resource overview, with charts for RAM usage, available disk space,
CPU/GPU activity and your models on disk, under the **Inference service**
card that starts, restarts and stops the app-owned server on a downloaded
model. Downloads stay in Models; authentication, attach and runtime metrics
stay in Server Settings.

The disk-space ring shows the volume containing the configured model download
folder, not the size of model files. Used/reserved includes other files and APFS
allocations. Imported models and caches may be on other volumes. A metadata helper
reads capacity with a three-second deadline, once when Dashboard becomes active
and every minute while it stays active. Missing volumes and broken links remain
unavailable. No weights are read or directories sized for this chart. A local
privacy manifest declares the disk-capacity API's display-only usage.

The inventory chart separates models ready to serve (complete weight files), incomplete/metadata-only
folders and folders not yet checked in the current scan. These are discovered
folders, not a count of models verified to run. One cached snapshot per repository
is included. Partial and ongoing scans are labeled. Refresh updates storage and
inventory without starting a model or downloading anything.

Optional **Live activity** retains up to four hours of CPU, GPU and unified-memory
history. It is off by default; the preference persists locally. Once enabled,
sampling continues across pages and in the background while the app window is
open, with a five-second pause between samples and a two-second deadline per probe.
Turn Live activity off to stop recording. Closing the window stops sampling;
quitting clears the in-memory history. Sleep and interrupted probes leave gaps. CPU uses process averages
normalized by logical cores; memory uses active, wired and compressed pages.
GPU utilization is available only when IOAccelerator exposes it. These host-wide
estimates are separate from runtime metrics and do not establish model fit or
inference performance. OS memory pressure remains unknown until a public system
event arrives. Missing, stale and interrupted samples leave gaps instead of zeros.

RAM shows estimated GB used with the percentage and total memory alongside it.
Select **1 min, 5 min, 15 min, 30 min, 1 hour, or 4 hours** from the Dashboard
range menu; changing the range does not change sampling or discard history.
History retains at most 6,000 samples within four hours. Each plot renders at
most 600 original samples, preserving first/last and low/high values per time
bucket and keeping gaps between disconnected segments. Hover inspection uses
the original history. A newly opened app shows only the history collected so far. Each trend has a fixed 0–100% scale. Hover
over a plot to inspect an actual sample and its timestamp; empty intervals stay
unavailable. Source explanations are under **How these measurements work** on Dashboard. The
native Swift Charts plot uses SwiftUI's Metal-backed offscreen composition, while
labels and controls remain native. This is a rendering choice, not a measured
performance claim. Chart appearance adapts to light/dark mode, increased contrast
and reduced motion; metric names and symbols supplement color.

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
The app-owned running model is labeled explicitly. Catalog links an active
download to Downloads.
The app performs compatibility validation when the operator explicitly starts a
model. Models from other tools are not silently deleted or hidden.

Use Command-1 through Command-5 to navigate Dashboard, Models, Server Settings, Chat and
Settings; Command-F focuses model search, Command-R refreshes models,
Shift-Command-R starts or restarts the server and Shift-Command-Period stops it.
Command-Comma still opens inline Settings. Search, selection and chat drafts
survive page switches and main-window reopening for the current app session.
API credentials remain in memory, and starting/connecting never happens merely
because a page was opened. Attached services retain Disconnect-only ownership.
