use crate::app::{AppState, LoadState, MODEL_CATALOG, ServerUrlKind};
use serde_json::{Value, json};

pub fn index_html() -> &'static str {
    r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AX ENGINE MANAGER</title>
  <link rel="stylesheet" href="/assets/manager.css">
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <span class="brand-glyph">◈</span>
      <span class="brand-name">AX ENGINE</span>
      <span class="brand-tag">MANAGER</span>
    </div>
    <div class="topbar-center">
      <span id="sys-status" class="sys-status">INITIALIZING</span>
    </div>
    <div class="topbar-right">
      <div class="server-indicator">
        <span id="server-dot" class="dot dot-off"></span>
        <span id="server-label">OFFLINE</span>
      </div>
      <button id="refresh" type="button" class="btn-icon" title="Refresh">⟳</button>
    </div>
  </header>

  <div class="workspace">
    <aside class="sidebar">

      <div class="panel">
        <div class="panel-header">
          <span class="panel-icon">◈</span>
          <span class="panel-title">SERVER</span>
        </div>

        <div class="field-group">
          <label class="field-label">PORT</label>
          <input id="port" class="field-input" type="number" min="1" max="65535" value="8080">
        </div>

        <div class="field-group">
          <label class="field-label">MODEL DIR</label>
          <input id="model-dir" class="field-input" type="text" spellcheck="false" placeholder="/path/to/model">
        </div>

        <div class="server-controls">
          <button id="start-server" class="btn btn-green" type="button">START</button>
          <button id="stop-server" class="btn btn-red" type="button">STOP</button>
          <button id="restart-server" class="btn btn-ghost full-width" type="button">RESTART</button>
        </div>

        <p id="server-status" class="status-text">Stopped</p>
        <ul id="endpoint-list" class="endpoint-list"></ul>
      </div>

      <div class="panel">
        <div class="panel-header">
          <span class="panel-icon">◈</span>
          <span class="panel-title">MODELS</span>
        </div>

        <div class="field-group">
          <label class="field-label">TYPE</label>
          <select id="model-kind" class="field-select"></select>
        </div>
        <div class="field-group">
          <label class="field-label">FAMILY</label>
          <select id="family" class="field-select"></select>
        </div>
        <div class="field-group">
          <label class="field-label">SIZE</label>
          <select id="model" class="field-select"></select>
        </div>

        <div class="repo-row">
          <span class="field-label">REPO</span>
          <span id="repo-id" class="repo-id">—</span>
        </div>

        <button id="download" type="button" class="btn btn-accent full-width">DOWNLOAD</button>
        <p id="download-status" class="status-text">IDLE</p>

        <div id="downloaded-list"></div>
      </div>

    </aside>

    <section class="chat-area">
      <div class="chat-header">
        <span class="panel-icon">◈</span>
        <span class="panel-title">INTELLIGENCE CORE</span>
        <div class="chat-header-right">
          <span id="chat-conn" class="chat-conn offline">server offline</span>
          <button id="chat-clear" class="btn-icon" type="button" title="Clear chat">✕</button>
        </div>
      </div>

      <div id="chat-messages" class="chat-messages">
        <div class="chat-empty" id="chat-empty">
          <div class="chat-empty-glyph">◈</div>
          <p>Start the server, then send a message.</p>
        </div>
      </div>

      <div class="chat-input-area">
        <div class="chat-input-row">
          <span class="prompt-arrow">›</span>
          <textarea id="chat-input" class="chat-input" rows="1"
            placeholder="enter message…" spellcheck="false"></textarea>
          <button id="chat-send" type="button" class="btn btn-accent">SEND</button>
        </div>
        <div class="chat-hints">
          <span>ENTER to send · SHIFT+ENTER for newline</span>
          <span id="chat-model-label"></span>
        </div>
      </div>
    </section>
  </div>

  <script src="/assets/manager.js"></script>
</body>
</html>
"##
}

pub fn manager_css() -> &'static str {
    r##":root {
  --bg:           #050810;
  --panel:        #0c1121;
  --panel-alt:    #0f1428;
  --border:       rgba(0,180,255,0.10);
  --border-hi:    rgba(0,200,255,0.40);
  --accent:       #00d4ff;
  --accent-dim:   rgba(0,212,255,0.55);
  --green:        #00e87a;
  --green-dim:    rgba(0,232,122,0.25);
  --red:          #ff3355;
  --red-dim:      rgba(255,51,85,0.20);
  --amber:        #ffaa00;
  --text:         #7aaac4;
  --text-hi:      #d4eeff;
  --text-dim:     #2d4a62;
  --mono:         'SF Mono','Menlo','Courier New',monospace;
  --sans:         -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --radius:       5px;
}

*,*::before,*::after { box-sizing: border-box; margin: 0; padding: 0; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }

body {
  background: var(--bg);
  background-image:
    linear-gradient(rgba(0,180,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,180,255,0.025) 1px, transparent 1px);
  background-size: 32px 32px;
  color: var(--text);
  font-family: var(--sans);
  font-size: 13px;
  height: 100vh;
  display: grid;
  grid-template-rows: 52px 1fr;
  overflow: hidden;
}

body::after {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 3px,
    rgba(0,0,0,0.04) 3px,
    rgba(0,0,0,0.04) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* ── Topbar ──────────────────────────────────────────────────── */
.topbar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 0 20px;
  background: rgba(5,8,16,0.96);
  border-bottom: 1px solid var(--border-hi);
  box-shadow: 0 1px 24px rgba(0,180,255,0.08);
  position: relative;
  z-index: 10;
}

.brand {
  display: flex;
  align-items: center;
  gap: 9px;
  flex-shrink: 0;
}

.brand-glyph {
  color: var(--accent);
  font-size: 18px;
  text-shadow: 0 0 12px var(--accent);
}

.brand-name {
  font-family: var(--mono);
  font-size: 14px;
  font-weight: 700;
  color: var(--text-hi);
  letter-spacing: 3px;
}

.brand-tag {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 2px;
  color: var(--accent-dim);
  border: 1px solid rgba(0,212,255,0.25);
  padding: 2px 5px;
  border-radius: 3px;
}

.topbar-center {
  flex: 1;
  text-align: center;
}

.sys-status {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 1px;
  color: var(--text-dim);
  text-transform: uppercase;
}

.topbar-right {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-shrink: 0;
}

.server-indicator {
  display: flex;
  align-items: center;
  gap: 7px;
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 1px;
}

/* ── Dots ────────────────────────────────────────────────────── */
.dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  display: inline-block;
  flex-shrink: 0;
}
.dot-on {
  background: var(--green);
  box-shadow: 0 0 10px var(--green-dim);
  animation: dot-pulse 2s ease-in-out infinite;
}
.dot-off  { background: var(--text-dim); }
.dot-amber {
  background: var(--amber);
  box-shadow: 0 0 10px rgba(255,170,0,0.3);
  animation: dot-pulse 0.8s ease-in-out infinite;
}

@keyframes dot-pulse {
  0%,100% { opacity: 1; }
  50%      { opacity: 0.35; }
}

.text-green { color: var(--green); }

/* ── Shared button base ──────────────────────────────────────── */
.btn-icon {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
  cursor: pointer;
  font-size: 15px;
  width: 30px; height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: border-color 0.15s, color 0.15s;
}
.btn-icon:hover { border-color: var(--accent); color: var(--accent); }

.btn {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
  cursor: pointer;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1px;
  padding: 7px 12px;
  text-align: center;
  transition: all 0.15s;
  white-space: nowrap;
}
.btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
.btn:disabled { opacity: 0.30; cursor: not-allowed; }

.btn-accent {
  border-color: var(--accent-dim);
  color: var(--accent);
  background: rgba(0,212,255,0.04);
}
.btn-accent:hover:not(:disabled) {
  background: rgba(0,212,255,0.10);
  box-shadow: 0 0 14px rgba(0,212,255,0.18);
}

.btn-green {
  border-color: rgba(0,232,122,0.4);
  color: var(--green);
  background: rgba(0,232,122,0.04);
}
.btn-green:hover:not(:disabled) {
  background: rgba(0,232,122,0.10);
  box-shadow: 0 0 10px rgba(0,232,122,0.18);
}

.btn-red {
  border-color: rgba(255,51,85,0.35);
  color: var(--red);
  background: rgba(255,51,85,0.04);
}
.btn-red:hover:not(:disabled) {
  background: rgba(255,51,85,0.10);
  box-shadow: 0 0 10px rgba(255,51,85,0.18);
}

.btn-ghost {
  border-color: var(--border);
  color: var(--text);
}

.full-width { width: 100%; margin-top: 6px; }

/* ── Workspace ───────────────────────────────────────────────── */
.workspace {
  display: grid;
  grid-template-columns: 272px 1fr;
  overflow: hidden;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
.sidebar {
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* ── Panels ──────────────────────────────────────────────────── */
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 13px;
  position: relative;
  overflow: hidden;
}
.panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent-dim), transparent);
}

.panel-header {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 13px;
}
.panel-icon {
  color: var(--accent);
  font-size: 11px;
  text-shadow: 0 0 8px var(--accent);
}
.panel-title {
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 2px;
  color: var(--accent);
}

/* ── Form fields ─────────────────────────────────────────────── */
.field-group { margin-bottom: 9px; }

.field-label {
  display: block;
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 1px;
  color: var(--text-dim);
  text-transform: uppercase;
  margin-bottom: 4px;
}

.field-input,
.field-select {
  width: 100%;
  background: var(--panel-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text-hi);
  font-family: var(--mono);
  font-size: 12px;
  padding: 6px 8px;
  outline: none;
  transition: border-color 0.15s;
  appearance: none;
  -webkit-appearance: none;
}
.field-input:focus,
.field-select:focus { border-color: var(--accent-dim); }

.field-select {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%2300d4ff'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 8px center;
  padding-right: 24px;
}
.field-select option { background: #0c1121; color: #d4eeff; }

/* ── Server controls ─────────────────────────────────────────── */
.server-controls {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  margin-bottom: 0;
}

.status-text {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-dim);
  margin: 7px 0 4px;
  min-height: 14px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.endpoint-list {
  list-style: none;
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.endpoint-list a {
  color: rgba(0,212,255,0.5);
  font-family: var(--mono);
  font-size: 9px;
  text-decoration: none;
  word-break: break-all;
  transition: color 0.15s;
}
.endpoint-list a:hover { color: var(--accent); }

/* ── Repo row ────────────────────────────────────────────────── */
.repo-row {
  display: flex;
  align-items: flex-start;
  gap: 7px;
  margin: 9px 0 12px;
}
.repo-id {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-hi);
  word-break: break-all;
  flex: 1;
}

/* ── Downloaded list ─────────────────────────────────────────── */
.dl-section-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 1px;
  color: var(--text-dim);
  margin: 12px 0 5px;
  text-transform: uppercase;
}
.dl-item {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 5px 0;
  border-bottom: 1px solid var(--border);
}
.dl-check { color: var(--green); font-size: 11px; flex-shrink: 0; }
.dl-repo {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--text);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.btn-xs {
  font-size: 9px;
  padding: 2px 6px;
  letter-spacing: 0;
  flex-shrink: 0;
}

/* ── Chat area ───────────────────────────────────────────────── */
.chat-area {
  display: grid;
  grid-template-rows: 44px 1fr auto;
  overflow: hidden;
}

.chat-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 18px;
  border-bottom: 1px solid var(--border);
  background: rgba(12,17,33,0.7);
}

.chat-header-right {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-left: auto;
}

.chat-conn {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.5px;
  transition: color 0.3s;
}
.chat-conn.online  { color: var(--green); }
.chat-conn.offline { color: var(--text-dim); }

/* ── Chat messages ───────────────────────────────────────────── */
.chat-messages {
  overflow-y: auto;
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 22px;
}

.chat-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  min-height: 200px;
  gap: 12px;
  color: var(--text-dim);
  user-select: none;
}
.chat-empty-glyph {
  font-size: 36px;
  color: rgba(0,212,255,0.12);
  text-shadow: 0 0 20px rgba(0,212,255,0.2);
  animation: glyph-float 4s ease-in-out infinite;
}
@keyframes glyph-float {
  0%,100% { transform: translateY(0); opacity: 0.6; }
  50%      { transform: translateY(-6px); opacity: 1; }
}
.chat-empty p {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 1px;
}

.chat-msg {
  display: flex;
  gap: 14px;
  animation: msg-in 0.18s ease-out;
}
@keyframes msg-in {
  from { opacity: 0; transform: translateY(5px); }
  to   { opacity: 1; transform: translateY(0); }
}

.msg-role {
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  padding-top: 3px;
  flex-shrink: 0;
  width: 28px;
  text-align: right;
}
.chat-msg-user      .msg-role { color: var(--accent); }
.chat-msg-assistant .msg-role { color: var(--green); }

.msg-body {
  font-size: 14px;
  line-height: 1.65;
  color: var(--text-hi);
  white-space: pre-wrap;
  word-break: break-word;
  flex: 1;
}
.chat-msg-user .msg-body { color: var(--text); }

/* Inline code */
.msg-body code {
  font-family: var(--mono);
  font-size: 12px;
  background: rgba(0,212,255,0.06);
  border: 1px solid rgba(0,212,255,0.15);
  border-radius: 3px;
  padding: 1px 5px;
  color: var(--accent);
}

/* Code block */
.msg-body pre {
  background: rgba(0,0,0,0.4);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 10px 12px;
  margin: 8px 0;
  overflow-x: auto;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text-hi);
  line-height: 1.5;
}
.msg-body pre code {
  background: none;
  border: none;
  padding: 0;
  color: inherit;
}

/* Streaming cursor */
.cursor {
  display: inline-block;
  color: var(--accent);
  animation: cur-blink 0.75s step-end infinite;
}
@keyframes cur-blink {
  0%,100% { opacity: 1; }
  50%      { opacity: 0; }
}

/* ── Chat input ──────────────────────────────────────────────── */
.chat-input-area {
  border-top: 1px solid var(--border);
  padding: 12px 18px;
  background: var(--panel);
}
.chat-input-row {
  display: flex;
  align-items: flex-end;
  gap: 9px;
  margin-bottom: 5px;
}
.prompt-arrow {
  font-family: var(--mono);
  font-size: 20px;
  color: var(--accent);
  padding-bottom: 7px;
  flex-shrink: 0;
  text-shadow: 0 0 8px var(--accent);
}
.chat-input {
  flex: 1;
  background: var(--panel-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text-hi);
  font-family: var(--sans);
  font-size: 14px;
  line-height: 1.5;
  padding: 8px 11px;
  resize: none;
  outline: none;
  transition: border-color 0.15s;
  min-height: 38px;
  max-height: 140px;
  overflow-y: auto;
}
.chat-input:focus { border-color: var(--accent-dim); }
.chat-input::placeholder { color: var(--text-dim); }

.chat-hints {
  display: flex;
  justify-content: space-between;
  font-family: var(--mono);
  font-size: 9px;
  color: var(--text-dim);
  letter-spacing: 0.5px;
}

/* ── Mobile ──────────────────────────────────────────────────── */
@media (max-width: 760px) {
  .workspace {
    grid-template-columns: 1fr;
    grid-template-rows: auto 1fr;
  }
  .sidebar {
    border-right: none;
    border-bottom: 1px solid var(--border);
    max-height: 45vh;
  }
  body { overflow: auto; }
}
"##
}

pub fn manager_js() -> &'static str {
    r##"
// ── Helpers ──────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

async function api(path, init = {}) {
  const res = await fetch(path, {
    headers: { 'content-type': 'application/json', ...(init.headers || {}) },
    ...init,
  });
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = {}; }
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

// ── App state ─────────────────────────────────────────────────────────────────

const app = {
  catalog: [],
  serverRunning: false,
  serverPort: 8080,
  downloadedModels: [],
  streaming: false,
};

// ── State rendering ───────────────────────────────────────────────────────────

function applyState(data) {
  app.catalog         = data.catalog || [];
  app.serverRunning   = !!(data.server && data.server.running);
  app.serverPort      = (data.server && data.server.port) || 8080;
  app.downloadedModels = data.downloaded_models || [];

  $('sys-status').textContent = data.status || '';

  // Server indicator dot
  const dot   = $('server-dot');
  const label = $('server-label');
  if (app.serverRunning) {
    dot.className   = 'dot dot-on';
    label.textContent = 'ONLINE';
    label.className   = 'text-green';
  } else {
    dot.className   = 'dot dot-off';
    label.textContent = 'OFFLINE';
    label.className   = '';
  }

  // Port / model-dir: only update when user is not focused on field
  if (document.activeElement !== $('port') && data.server) {
    $('port').value = data.server.port;
  }
  if (!$('model-dir').value && data.model_dir) {
    $('model-dir').value = data.model_dir;
  }

  $('server-status').textContent = (data.server && data.server.status) || '';
  renderEndpoints((data.server && data.server.endpoints) || []);
  $('download-status').textContent = data.download_status || 'IDLE';

  fillModelSelectors(data);
  renderDownloaded(app.downloadedModels);

  // Chat connection badge
  const conn = $('chat-conn');
  if (app.serverRunning) {
    conn.textContent  = `→ 127.0.0.1:${app.serverPort}`;
    conn.className    = 'chat-conn online';
  } else {
    conn.textContent  = 'server offline';
    conn.className    = 'chat-conn offline';
  }

  $('chat-model-label').textContent =
    data.model_dir ? data.model_dir.split('/').slice(-2).join('/') : '';
}

// ── Model selectors ───────────────────────────────────────────────────────────

function kinds()            { return [...new Set(app.catalog.map(e => e.kind))]; }
function familiesOf(kind)   { return [...new Set(app.catalog.filter(e => e.kind === kind).map(e => e.family))]; }
function modelsOf(kind, fam){ return app.catalog.filter(e => e.kind === kind && e.family === fam); }

function fillModelSelectors(data) {
  const prevKind = $('model-kind').value;
  const prevFam  = $('family').value;
  const prevRepo = $('model').value;

  const ks = kinds();
  $('model-kind').innerHTML = ks.map(k => `<option value="${k}">${k}</option>`).join('');
  const kind = ks.includes(prevKind) ? prevKind : ks[0];
  $('model-kind').value = kind;

  const fams = familiesOf(kind);
  $('family').innerHTML = fams.map(f => `<option value="${f}">${f}</option>`).join('');
  const fam = fams.includes(prevFam) ? prevFam : fams[0];
  $('family').value = fam;

  fillSizes(kind, fam, prevRepo || (data && data.selected_repo_id));
}

function fillSizes(kind, fam, preferred) {
  const models = modelsOf(kind, fam);
  $('model').innerHTML = models.map(m =>
    `<option value="${m.repo_id}">${m.label} · ${m.note}</option>`
  ).join('');
  if (preferred && models.some(m => m.repo_id === preferred)) {
    $('model').value = preferred;
  }
  updateRepoLabel();
}

function updateRepoLabel() {
  $('repo-id').textContent = $('model').value || '—';
}

// ── Endpoints ─────────────────────────────────────────────────────────────────

function renderEndpoints(eps) {
  $('endpoint-list').innerHTML = eps.slice(0, 4).map(ep =>
    `<li><a href="${ep.url}" target="_blank" rel="noopener">${ep.label}</a></li>`
  ).join('');
}

// ── Downloaded list ───────────────────────────────────────────────────────────

function renderDownloaded(models) {
  const el = $('downloaded-list');
  if (!models.length) { el.innerHTML = ''; return; }
  el.innerHTML =
    `<div class="dl-section-label">DOWNLOADED</div>` +
    models.map(m => `
      <div class="dl-item">
        <span class="dl-check">✓</span>
        <span class="dl-repo" title="${m.path || ''}">${m.repo_id}</span>
        <button class="btn btn-ghost btn-xs dl-use"
          data-path="${m.path || ''}"
          data-repo="${m.repo_id}">USE</button>
      </div>`).join('');
  el.querySelectorAll('.dl-use').forEach(btn => {
    btn.addEventListener('click', () => { $('model-dir').value = btn.dataset.path; });
  });
}

// ── Server controls ───────────────────────────────────────────────────────────

function serverPayload() {
  return JSON.stringify({
    port:      Number($('port').value || 8080),
    model_dir: $('model-dir').value.trim(),
  });
}

function setServerBtns(busy) {
  ['start-server','stop-server','restart-server'].forEach(id => {
    $(id).disabled = busy;
  });
}

async function startServer() {
  setServerBtns(true);
  $('server-dot').className = 'dot dot-amber';
  $('server-status').textContent = 'Starting…';
  try {
    await api('/api/server/start', { method: 'POST', body: serverPayload() });
  } catch (err) {
    $('server-status').textContent = err.message;
  }
  await loadState();
  setServerBtns(false);
}

async function stopServer() {
  setServerBtns(true);
  try {
    await api('/api/server/stop', { method: 'POST', body: '{}' });
  } catch (err) {
    $('server-status').textContent = err.message;
  }
  await loadState();
  setServerBtns(false);
}

async function restartServer() {
  setServerBtns(true);
  $('server-dot').className = 'dot dot-amber';
  $('server-status').textContent = 'Restarting…';
  try {
    await api('/api/server/restart', { method: 'POST', body: serverPayload() });
  } catch (err) {
    $('server-status').textContent = err.message;
  }
  await loadState();
  setServerBtns(false);
}

// ── Download ──────────────────────────────────────────────────────────────────

async function startDownload() {
  const entry = app.catalog.find(e => e.repo_id === $('model').value);
  if (!entry) return;
  $('download').disabled = true;
  $('download-status').textContent = `Downloading ${entry.repo_id}…`;
  try {
    const job = await api('/api/download', {
      method: 'POST',
      body: JSON.stringify({ repo_id: entry.repo_id, kind: entry.kind }),
    });
    pollDownload(job.id, entry.repo_id);
  } catch (err) {
    $('download-status').textContent = err.message;
    $('download').disabled = false;
  }
}

async function pollDownload(jobId, repoId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    await loadState();
    if (job.status === 'running') {
      setTimeout(() => pollDownload(jobId, repoId), 1500);
    } else {
      $('download').disabled = false;
      $('download-status').textContent =
        job.status === 'succeeded' ? `✓ ${repoId}` : `✗ ${job.message || 'failed'}`;
    }
  } catch (err) {
    $('download').disabled = false;
    $('download-status').textContent = err.message;
  }
}

// ── Chat ──────────────────────────────────────────────────────────────────────

const history = [];  // {role, content}

function ensureChatVisible() {
  const empty = $('chat-empty');
  if (empty) empty.remove();
}

function renderMarkdown(text) {
  // Simple inline markdown: code blocks, inline code, bold
  const escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  return escaped
    // fenced code blocks (```lang\n...\n```)
    .replace(/```[\w]*\n([\s\S]*?)```/g, (_, code) =>
      `<pre><code>${code.trimEnd()}</code></pre>`)
    // inline code
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    // bold
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    // newlines (outside pre)
    .replace(/\n/g, '<br>');
}

function addMsgEl(role, content, streaming) {
  ensureChatVisible();
  const wrap = document.createElement('div');
  wrap.className = `chat-msg chat-msg-${role}`;

  const lbl = document.createElement('span');
  lbl.className = 'msg-role';
  lbl.textContent = role === 'user' ? 'YOU' : 'AX';

  const body = document.createElement('div');
  body.className = 'msg-body';

  if (streaming) {
    body.appendChild(makeCursor());
  } else if (role === 'assistant') {
    body.innerHTML = renderMarkdown(content);
  } else {
    body.textContent = content;
  }

  wrap.append(lbl, body);
  $('chat-messages').append(wrap);
  scrollChat();
  return body;
}

function makeCursor() {
  const span = document.createElement('span');
  span.className = 'cursor';
  span.textContent = '▊';
  return span;
}

function appendToken(bodyEl, token) {
  const cursor = bodyEl.querySelector('.cursor');
  if (cursor) cursor.remove();
  // append as text node so partial markdown is safe
  bodyEl.insertAdjacentText('beforeend', token);
  bodyEl.appendChild(makeCursor());
  scrollChat();
}

function finalizeBody(bodyEl, fullText) {
  const cursor = bodyEl.querySelector('.cursor');
  if (cursor) cursor.remove();
  bodyEl.innerHTML = renderMarkdown(fullText);
  scrollChat();
}

function scrollChat() {
  const el = $('chat-messages');
  el.scrollTop = el.scrollHeight;
}

async function sendMessage() {
  const input  = $('chat-input');
  const text   = input.value.trim();
  if (!text || app.streaming) return;

  input.value = '';
  autoResize(input);

  history.push({ role: 'user', content: text });
  addMsgEl('user', text, false);

  const bodyEl = addMsgEl('assistant', '', true);
  app.streaming = true;
  $('chat-send').disabled = true;
  $('chat-send').textContent = '···';

  let accumulated = '';

  try {
    const res = await fetch('/api/proxy/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        messages:   history.slice(),
        stream:     true,
        max_tokens: 2048,
      }),
    });

    if (!res.ok) {
      let msg;
      try { msg = (await res.json()).error; } catch { msg = res.statusText; }
      throw new Error(msg || `HTTP ${res.status}`);
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let sseBuffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      sseBuffer += decoder.decode(value, { stream: true });

      let nl;
      while ((nl = sseBuffer.indexOf('\n')) !== -1) {
        const line = sseBuffer.slice(0, nl).trim();
        sseBuffer  = sseBuffer.slice(nl + 1);
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') break;
        let chunk;
        try { chunk = JSON.parse(payload); } catch { continue; }
        const delta = (chunk.choices && chunk.choices[0] && chunk.choices[0].delta && chunk.choices[0].delta.content) || '';
        if (delta) {
          accumulated += delta;
          appendToken(bodyEl, delta);
        }
      }
    }
  } catch (err) {
    accumulated = `[error: ${err.message}]`;
  }

  finalizeBody(bodyEl, accumulated || '(empty response)');
  history.push({ role: 'assistant', content: accumulated });

  app.streaming = false;
  $('chat-send').disabled = false;
  $('chat-send').textContent = 'SEND';
}

function clearChat() {
  history.length = 0;
  const msgs = $('chat-messages');
  msgs.innerHTML = '';
  const empty = document.createElement('div');
  empty.className = 'chat-empty';
  empty.id = 'chat-empty';
  empty.innerHTML = '<div class="chat-empty-glyph">◈</div><p>Start the server, then send a message.</p>';
  msgs.appendChild(empty);
}

// ── Textarea auto-resize ──────────────────────────────────────────────────────

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

// ── Load state ────────────────────────────────────────────────────────────────

async function loadState() {
  try {
    const data = await api('/api/state');
    applyState(data);
  } catch (err) {
    $('sys-status').textContent = err.message;
  }
}

// ── Wire events ───────────────────────────────────────────────────────────────

$('model-kind').addEventListener('change', () => {
  const kind = $('model-kind').value;
  const fams = familiesOf(kind);
  $('family').innerHTML = fams.map(f => `<option value="${f}">${f}</option>`).join('');
  fillSizes(kind, fams[0], null);
});

$('family').addEventListener('change', () => {
  fillSizes($('model-kind').value, $('family').value, null);
});

$('model').addEventListener('change', updateRepoLabel);

$('download').addEventListener('click', startDownload);
$('start-server').addEventListener('click', startServer);
$('stop-server').addEventListener('click', stopServer);
$('restart-server').addEventListener('click', restartServer);
$('refresh').addEventListener('click', loadState);
$('chat-clear').addEventListener('click', clearChat);
$('chat-send').addEventListener('click', sendMessage);

$('chat-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

$('chat-input').addEventListener('input', function () { autoResize(this); });

// ── Boot ──────────────────────────────────────────────────────────────────────

loadState().catch(err => {
  $('sys-status').textContent = String(err.message || err);
});
"##
}

pub fn catalog_json() -> Vec<Value> {
    MODEL_CATALOG
        .iter()
        .map(|entry| {
            json!({
                "kind":    entry.kind.label(),
                "family":  entry.family.label(),
                "label":   entry.label,
                "repo_id": entry.repo_id,
                "note":    entry.note,
            })
        })
        .collect()
}

pub fn readiness_json(state: &AppState) -> Value {
    json!({
        "doctor":          load_state_label(&state.doctor),
        "workflow":        workflow_label(state),
        "model_artifacts": model_artifacts_label(state),
        "benchmark":       load_state_label(&state.benchmark_summary),
        "artifacts":       load_state_label(&state.artifacts),
    })
}

pub fn server_endpoints(base_url: &str) -> Vec<Value> {
    ServerUrlKind::ALL
        .into_iter()
        .map(|kind| {
            json!({
                "label": kind.label(),
                "url":   format!("{}{}", base_url.trim_end_matches('/'), kind.path()),
            })
        })
        .collect()
}

pub fn current_model_dir(state: &AppState) -> Option<String> {
    match &state.doctor {
        LoadState::Ready(report) => report.model_artifacts.path.clone(),
        _ => None,
    }
}

fn workflow_label(state: &AppState) -> String {
    match &state.doctor {
        LoadState::Ready(report) => report.workflow.mode.clone(),
        LoadState::Unavailable(m) | LoadState::NotLoaded(m) => m.clone(),
    }
}

fn model_artifacts_label(state: &AppState) -> String {
    match &state.doctor {
        LoadState::Ready(report) => report.model_artifacts.status.clone(),
        LoadState::Unavailable(m) | LoadState::NotLoaded(m) => m.clone(),
    }
}

fn load_state_label<T>(state: &LoadState<T>) -> &'static str {
    match state {
        LoadState::Ready(_)       => "ready",
        LoadState::Unavailable(_) => "unavailable",
        LoadState::NotLoaded(_)   => "not_loaded",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_assets_include_manager_mount_points() {
        assert!(index_html().contains("AX ENGINE"));
        assert!(manager_js().contains("/api/download"));
        assert!(manager_css().contains(".chat-area"));
    }

    #[test]
    fn catalog_json_exposes_repo_ids() {
        let catalog = catalog_json();
        assert!(
            catalog
                .iter()
                .any(|entry| entry["repo_id"] == "mlx-community/Qwen3-4B-4bit")
        );
    }
}
