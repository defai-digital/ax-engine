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

**Layout.** The sidebar holds **Home** (Command-1), **Playground**
(Command-2), **Models** (Command-3), **API & Logs** (Command-4) and **System**
(Command-5), then **Pinned** models (or, before you pin one, your **Recent**
ones): starred models that serve in one click
(while a server runs, a pinned model is selected on Home, where Restart
switches to it). The toolbar carries one **server capsule** on every page:
state, model, endpoint, live token rate, a copy-URL button and the single
lifecycle action that applies (Start, Stop, Cancel or Disconnect). Green
means this app's own server is serving; blue marks an attached service the
app did not start, and speculative-decoding readings; indigo is the accent
for actions and selection. The toolbar also has a **Downloads** popover and
the **command palette** (Command-K), which jumps to a page, serves a
downloaded model or runs a server action through the same gates as the
buttons. **Settings** is a standard window (Command-Comma). The sidebar header
shows the app version, chip and unified memory of this Mac.

**First run.** When a complete library scan finds no models, a short guide
opens: what AX Engine does, a list of catalog models (download size within
this Mac's memory first, smallest first; a size comparison, not a fit
claim), the download with live progress, and **Start Server**. Closing it
keeps any download running. It is offered once; reopen it from **Help ›
Getting Started**.

1. **Start serving (Home):** Home opens with one hero panel for the selected
   model: its state, short title (for example **Tiel Coder 35B A3B**) with
   quantization, MTP and size tags, the local endpoint with a copy button and
   **Start Server** (or Shift-Command-R, the toolbar capsule or the menu-bar
   item). **Choose Model** opens a searchable picker that groups models under
   **Favorites** (star a row to pin it), **Recent** (the last five models you
   started) and the rest; arrow keys and Return work from the search field,
   and **Download more models** opens the Catalog. With nothing downloaded
   yet, the panel offers **Download a Model**. With nothing chosen, the
   running model, then the most recently started one, a favorite or the first
   download is preselected. While starting it shows the elapsed time and the
   runtime's latest output line, and **Cancel Start** stops the launch. Once
   running it offers **Open Playground**, **Switch Model** and **Restart**.
   By default the server needs no API key: it listens on 127.0.0.1 only, so
   local clients can call it directly. The right side of the hero is **Token
   speed · live**: tokens the runtime scheduled per second (prompt and output
   together) between two-second polls of `/metrics`, with a five-minute
   sparkline, the last Playground reply's first-text time and prefill rate,
   interval MTP draft acceptance and requests in flight. Below it,
   **Unified memory** splits the runtime's MLX active and cache bytes from
   other apps (from the host monitor), **Throughput** charts scheduled tokens
   per 10 seconds, **Requests** shows the runtime's HTTP counters (generating
   now, completed, HTTP requests, 4xx/5xx errors) since the server started,
   and **Connect an app** repeats the setups from API & Logs. The runtime does
   not report per-request or per-client history, so Home shows none. Missing
   counters show a dash; none of these is a benchmark.
2. **API & Logs:** **Endpoint** lists the OpenAI base URL, the Anthropic base
   URL and the served model ID, each with a copy button, followed by the API
   key setting, **Serving profile**, **Parallel requests** and the port.
   **Connect an app** has ready-to-paste setups for **Claude Code** (exports
   `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL` and `ANTHROPIC_AUTH_TOKEN`, then
   runs `claude`), **Python** (OpenAI SDK), **TypeScript**, **curl** and
   **Env vars**; when a key is required they reference `$AX_ENGINE_API_KEY`
   and never contain the key. **Logs** shows the runtime output with
   errors and warnings coloured, **All** or **Problems**, a text filter,
   **Follow** to stay at the newest line and **Copy**. **Require an API key**
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
   key save as soon as they change, on Home, API & Logs or Settings
   › Server, without touching other unsaved Settings edits, and apply at the
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
   and Restart ask for confirmation only while a reply is being generated;
   Stop then offers **Keep Serving**, **Finish Reply, Then Stop** (waits up to
   10 minutes for streams to end; choosing Keep Serving later cancels it) and
   **Stop Now**. Stop gives the runtime eight seconds to drain before it is
   force-killed. When the runtime exits on its own (a signal such as SIGKILL
   is named as one), Home shows its last problem lines with **Start Again**, **Copy Report** and **Open Logs**, plus
   a memory hint when the Mac is above 70% in use. When the port is taken,
   Home offers **Attach** (use that server; the app never stops it) or
   **Change Port…**. A memory-pressure banner appears on Home and System only
   while macOS reports Warning or Critical pressure. Logs can be filtered by text, narrowed to
   warnings and errors, and limited to the last 100 or 500 lines; filters
   change the view only, and **Copy** copies the whole bounded output. Runtime metrics
   and history appear only while something is serving; Mac CPU, GPU and memory
   charts are on System. **Already running AX Engine on this Mac? Connect to
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
3. **Models:** three segments. **Your Models** is a table of app-managed and
   cached models. The **AutomatosX | All models** control at the top of the
   table chooses which ones are listed: AutomatosX packs only (the default),
   or every model folder the library scan found on this Mac, including
   imported folders and other publishers' repositories in the Hugging Face
   cache. The choice is remembered and applies to every chooser (Home, the
   command palette, A/B compare and the sidebar). Rows from other publishers
   carry an **Unofficial** tag (not officially supported): the app can try to
   serve them, and the runtime checks compatibility at start and may refuse
   them. The app removes only folders it downloaded itself and AutomatosX
   repositories in the Hugging Face cache. Switching back to AutomatosX keeps the model being
   served listed; a hidden selection falls back to an AutomatosX model and
   pinned models return with the wider scope. The table footer says how many
   models the narrow scope leaves out. The same control is on the Catalog. Each row shows name and source, format, size and status:
   **Ready**, **Incomplete** or **Not checked**, plus **Selected** and
   **Serving**), ready models first. Selecting a row opens an inspector with
   its tags, **Serve** and **Pin**, size on disk against unified memory (a
   size comparison, not a fit claim), repository, revision, source, file
   completeness and location, **Show in Finder**, **Copy Path** and
   **Remove…** under the same gates as before. In narrow windows the Format
   column is hidden, then the inspector moves below the table. One snapshot per cached repository is
   shown, preferring complete weights before the newest revision. **Serve**
   on a ready row starts the server on it when nothing is serving; while a
   server runs, **Select on Home** selects it there so Restart switches
   to it. The same action is on downloaded Catalog rows and finished
   downloads. **Catalog** shows public AutomatosX packs (no Hugging Face
   account or token) as cards grouped into those that fit this Mac's unified memory and
   a collapsed group of larger ones; each row shows the runtime's approximate
   pack size, whether MTP artifacts are included and its certification word
   when recorded, plus the free space on the model volume. These are size
   comparisons, not fit or qualification claims. A pack whose estimate exceeds
   the free space asks first: AX Engine checks the exact remaining size before
   downloading and stops without touching existing models when it does not
   fit, so **Download Anyway** is safe and a resumed download may need less.
   Under **All models** the Catalog adds **Community models · not officially
   supported**: other publishers' public repositories that the runtime has a
   serve profile for (`community_targets` in `download --list --json`). They
   carry the **Unofficial** tag, the first such download asks once for
   confirmation, and they use the same anonymous, checksum-verified download
   path. AutomatosX has not tested them; the runtime may refuse one at start.
   A runtime that predates `community_targets` lists none. Repositories the
   runtime does not list are not offered in the GUI: download them with the
   CLI, after which they appear under **Your Models › All models**.
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
   set the generation profile, a **System prompt** and see the sampling and
   maximum-token settings from chips under the composer,
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
   Markdown headings render as headings and code blocks show their language.
   The **Telemetry** inspector (toolbar button or Option-Command-I) shows the
   live token rate, active requests and MTP draft acceptance; for the current
   or last reply the first-text time, elapsed time, native decode and prefill
   rates and token counts; prefix-cache hit rate and reused tokens since the
   server started; and the model context and serving mode.

5. **Server metrics:** inspect live runtime counters on API & Logs while a
   server runs. Missing measurements appear as unavailable. Counters are not a
   benchmark or a throughput certification.
6. **Menu bar and Server menu:** the menu-bar item (on by default; Settings ›
   General) shows the server state, with the live token rate next to its icon
   while one is reported. Its panel shows the state and address, the model with
   its tags and uptime, the live token rate with a sparkline and MTP draft
   acceptance, Start, Stop or Restart, a **Switch model** list of up to five
   downloaded models, Copy OpenAI base URL, Open Playground, Open AX Engine
   and Quit. The **Server** menu has Start/Restart Server
   (Shift-Command-R), Stop Server (Shift-Command-Period), Copy API URL and
   Open Playground.

7. **Compare A/B:** Playground's **Compare A/B** mode runs one prompt on the
   serving model (A), restarts the app-owned server on model B, runs it again
   and, with **Return to A**, restarts on A. Both runs use the same settings,
   snapshotted at the start. It asks first, because each restart reloads a
   model and cuts off other clients. Each side shows decode rate, time to
   first text, MTP draft acceptance (from server-wide counters, so other
   traffic during the run counts) and total time; a token-over-time chart and
   result rows appear after both runs finish. It is one run per side, not a
   benchmark. Starting, stopping or switching the server during a compare
   cancels it and skips the return to A.

Connect attaches to an existing AX Engine service on the configured local port.
Its API key stays in memory. Disconnect and app quit do not stop an attached
service. Quit asks before stopping app-owned inference or downloads; with the
menu-bar item on it also offers **Keep Serving in Menu Bar**. The dialog quits
on its own after 30 seconds, stopping the server and freeing its memory, so
an unattended quit (logout, a script) does not hold the model; choose
**Cancel** within that time to keep running. Closing a
window follows macOS app conventions; use Quit to stop app-owned work.

Source builds (`scripts/build-macos-app.sh` without `--distribution`) accept
QA launch arguments for screenshots and end-to-end checks:
`-AXOpenPage home|chat|models|api|system`, `-AXOpenSettings YES`,
`-AXSettingsPane <tab>`, `-AXStartServer YES`, `-AXShowMenuPanel YES`,
`-AXFinishThenStop YES` (chooses Finish Reply, Then Stop once a request
streams), `-AXSimulatePressure Warning|Critical` (a real trigger needs root),
and `-AXCompareModelB <path substring>` with `-AXComparePrompt <text>`.
Distribution builds ignore them.

## Settings navigation

Settings is a standard window (Command-Comma, the AX Engine menu or the
sidebar's Settings row). Its tabs are **General** (menu bar, updates,
privacy), **Server** (port, loopback listen address, serving mode, request
cap, API key), **Models** (download folder), **Playground** (request
profiles) and **Advanced** (runtime executable, version and open-source
licenses). Draft edits survive tab switches. Save validates and persists
them; Revert restores the saved values. Runtime and model operations lock
the runtime location, download folder and port while busy. Playground-only
changes remain editable and apply to the next Playground request without
restarting the server or rescanning models. Serving options save as soon
as they change.

## Playground profiles

Settings › Playground opens with **Coding**, **Chatbot**, **Media**, and **Custom**. Profiles
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

## System and discovery

System is the host overview: Memory, GPU and CPU charts, a **Model storage**
panel and a facts strip (chip, memory, macOS, AX Engine version, models ready
to serve and memory pressure) with **Copy diagnostics**. Serving lives on Home and in the
toolbar capsule; downloads stay in Models; authentication, attach and runtime
metrics stay in API & Logs.

Model storage shows the volume containing the configured model download
folder: model bytes from the library index, other used space in gray and the
free space, plus the Hugging Face cache's model bytes. Used space includes
other files and APFS allocations. Imported models and caches may be on other volumes. A metadata helper
reads capacity with a three-second deadline, once when System becomes active
and every minute while it stays active. Missing volumes and broken links remain
unavailable. No weights are read or directories sized for this chart. A local
privacy manifest declares the disk-capacity API's display-only usage.

The facts strip counts models ready to serve (complete weight files); its
tooltip adds incomplete/metadata-only folders and folders not yet checked in
the current scan. These are discovered
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
Select **5m, 1h or 4h** from the System range control; changing the range does not change sampling or discard history.
History retains at most 6,000 samples within four hours. Each plot renders at
most 600 original samples, preserving first/last and low/high values per time
bucket and keeping gaps between disconnected segments. Hover inspection uses
the original history. A newly opened app shows only the history collected so far. Each trend has a fixed 0–100% scale. Hover
over a plot to inspect an actual sample and its timestamp; empty intervals stay
unavailable. Source explanations are under **How these numbers are measured** on System. The
native Swift Charts plot uses SwiftUI's Metal-backed offscreen composition, while
labels and controls remain native. This is a rendering choice, not a measured
performance claim. Chart appearance adapts to light/dark mode, increased contrast
and reduced motion; metric names and symbols supplement color.

API & Logs' collapsed **Runtime history** uses existing service polls for MLX active
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

Model rows show a compact name, source, format, size and status; full paths and
revision details are in the inspector. Complete files do not establish runtime
compatibility, and absence from the runtime catalog does not mean unsupported.
The filter field matches folder and repository names without claiming
compatibility. Search offers a clear recovery action when nothing matches.
The app-owned running model is labeled explicitly. Catalog links an active
download to Downloads.
The app performs compatibility validation when the operator explicitly starts a
model. Models from other tools are not silently deleted or hidden.

Use Command-1 through Command-5 to navigate Home, Playground, Models, API & Logs
and System; Command-K opens the command palette, Command-F focuses model
search, Command-R refreshes models, Shift-Command-R starts or restarts the
server, Shift-Command-Period stops it and Command-Period stops a reply.
Command-Comma opens the Settings window. Search, selection and chat drafts
survive page switches and main-window reopening for the current app session.
API credentials remain in memory, and starting/connecting never happens merely
because a page was opened. Attached services retain Disconnect-only ownership.
