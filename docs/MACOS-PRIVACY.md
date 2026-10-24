# AX Engine for macOS: privacy and network use

The native AX Engine app does not upload conversations, usage analytics or
telemetry to AutomatosX. It has no bundled analytics, advertising or tracking SDK.
The app's privacy manifest declares no collected data types and no tracking.

## Local processing and storage

Playground sends inference requests to the AX Engine service on this Mac using
loopback networking. The bundled native runtime starts with offline model loading.
A runtime you explicitly supply, or another client you use with AX Engine, has
its own behavior; this statement describes the built-in app and bundled runtime.

Settings and saved conversations are local JSON files in
`~/Library/Application Support/AX Engine/`. Conversation files include prompts,
responses, reasoning when returned by the model, and per-reply speed measurements.
They remain until replaced or cleared. **Clear** in Playground removes the app's
saved conversation; it does not manage copies you exported or system backups.
**Show local data folder** in Settings > Privacy opens the storage location.

CPU, GPU, memory and inference counters are read locally for the dashboard and
server views. Their bounded activity histories stay in memory and reset when the
app quits. They are not an analytics upload. Runtime diagnostics are displayed
locally; diagnostic output can contain model names and local filesystem paths.

## User-requested network activity

Model downloads contact Hugging Face and its delivery hosts. Those services see
ordinary download-request information such as the requested model files and the
connection's IP address. Chat contents are not part of model download requests.
Opening a source-repository link launches the user's browser. These destinations
have their own privacy practices. Installing or updating the app also contacts
the distribution service used by the chosen installer.

No-telemetry behavior does not mean that downloads and links operate without a
network connection. macOS diagnostics and operating-system sharing choices are
controlled separately in System Settings.

## Disclosure design

Playground includes a short no-telemetry notice near the message field.
The Settings > Privacy tab provides the local-storage, retention and network details without a consent popup
or a telemetry switch for a feature that does not exist. Model download screens
identify the network destination before downloading.

References: [Apple App Privacy Details](https://developer.apple.com/app-store/app-privacy-details/)
and [User Privacy and Data Use](https://developer.apple.com/app-store/user-privacy-and-data-use/).
