use crate::app::{AppState, LoadState, MODEL_CATALOG, ServerUrlKind};
use serde_json::{Value, json};

const LOGO_B64: &str = include_str!("logo_b64.txt");

pub fn index_html() -> String {
    let logo_src = format!("data:image/png;base64,{LOGO_B64}");
    format!(
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
      <img src="{logo_src}" class="brand-logo" alt="AX Engine">
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

      <!-- STEP 1: SELECT MODEL -->
      <div class="panel">
        <div class="panel-header">
          <span class="step-badge">1</span>
          <span class="panel-title">SELECT MODEL</span>
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

        <div id="model-status" class="model-status"></div>

        <button id="download" type="button" class="btn btn-accent full-width">DOWNLOAD</button>
        <div id="dl-progress-wrap" class="dl-progress-wrap" hidden>
          <div id="dl-progress-fill" class="dl-progress-fill"></div>
        </div>
        <p id="download-status" class="status-text"></p>

        <div id="downloaded-list"></div>
      </div>

      <!-- STEP 2: LAUNCH SERVER -->
      <div class="panel">
        <div class="panel-header">
          <span class="step-badge">2</span>
          <span class="panel-title">LAUNCH SERVER</span>
        </div>

        <div class="field-group">
          <label class="field-label">MODEL PATH</label>
          <input id="model-dir" class="field-input" type="text" spellcheck="false"
            placeholder="select a downloaded model above">
        </div>
        <div class="field-group">
          <label class="field-label">PORT</label>
          <input id="port" class="field-input field-input-sm" type="number"
            min="1" max="65535" value="8080">
        </div>
        <div class="field-group">
          <label class="field-label">ENGINE</label>
          <select id="engine" class="field-select">
            <option value="ax-engine">ax-engine</option>
            <option value="ax-engine-ngram">ax-engine n-gram</option>
            <option value="mlx-lm">mlx-lm</option>
            <option value="mlx-swift">mlx-swift</option>
          </select>
        </div>

        <div class="server-controls">
          <button id="start-server" class="btn btn-green" type="button">▶ START</button>
          <button id="stop-server" class="btn btn-red" type="button">■ STOP</button>
          <button id="restart-server" class="btn btn-ghost full-width" type="button">↺ RESTART</button>
        </div>

        <p id="server-status" class="status-text">Stopped</p>
        <ul id="endpoint-list" class="endpoint-list"></ul>
      </div>

    </aside>

    <!-- STEP 3: CHAT -->
    <section class="chat-area">
      <div class="chat-header">
        <span class="step-badge">3</span>
        <span class="panel-title">CHAT</span>
        <div class="chat-header-right">
          <span id="chat-conn" class="chat-conn offline">server offline</span>
          <button id="chat-clear" class="btn-icon" type="button" title="Clear">✕</button>
        </div>
      </div>

      <div id="chat-messages" class="chat-messages">
        <div class="chat-empty" id="chat-empty">
          <img src="{logo_src}" class="chat-empty-logo" alt="AX Engine">
          <ol class="flow-steps">
            <li>Select a model above</li>
            <li>Download it if needed</li>
            <li>Start the server</li>
            <li>Chat here</li>
          </ol>
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
"##,
        logo_src = logo_src
    )
}

pub fn manager_css() -> &'static str {
    r##":root {
  --bg:        #050810;
  --panel:     #0c1121;
  --panel-alt: #0f1428;
  --border:    rgba(0,180,255,0.10);
  --border-hi: rgba(0,200,255,0.38);
  --accent:    #00d4ff;
  --accent-d:  rgba(0,212,255,0.52);
  --green:     #00e87a;
  --green-d:   rgba(0,232,122,0.22);
  --red:       #ff3355;
  --amber:     #ffaa00;
  --text:      #7aaac4;
  --text-hi:   #d4eeff;
  --text-dim:  #2b4560;
  --mono:      'SF Mono','Menlo','Courier New',monospace;
  --sans:      -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --r:         5px;
}

*,*::before,*::after { box-sizing: border-box; margin: 0; padding: 0; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }

body {
  background: var(--bg);
  background-image:
    linear-gradient(rgba(0,180,255,0.022) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,180,255,0.022) 1px,transparent 1px);
  background-size: 32px 32px;
  color: var(--text);
  font-family: var(--sans);
  font-size: 13px;
  height: 100vh;
  display: grid;
  grid-template-rows: 52px 1fr;
  overflow: hidden;
}

/* scanlines */
body::after {
  content:'';
  position:fixed;inset:0;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.04) 3px,rgba(0,0,0,0.04) 4px);
  pointer-events:none;z-index:9999;
}

/* ── Topbar ──────────────────────────────────────────────── */
.topbar {
  display:flex;align-items:center;gap:16px;
  padding:0 20px;
  background:rgba(5,8,16,0.96);
  border-bottom:1px solid var(--border-hi);
  box-shadow:0 1px 24px rgba(0,180,255,0.07);
  position:relative;z-index:10;
}
.brand { display:flex;align-items:center;gap:9px;flex-shrink:0; }
.brand-logo { width:28px;height:28px;object-fit:contain;filter:drop-shadow(0 0 6px rgba(0,180,255,0.55)); }
.brand-name  { font-family:var(--mono);font-size:14px;font-weight:700;color:var(--text-hi);letter-spacing:3px; }
.brand-tag   { font-family:var(--mono);font-size:9px;letter-spacing:2px;color:var(--accent-d);border:1px solid rgba(0,212,255,0.22);padding:2px 5px;border-radius:3px; }
.topbar-center { flex:1;text-align:center; }
.sys-status  { font-family:var(--mono);font-size:10px;letter-spacing:1px;color:var(--text-dim);text-transform:uppercase; }
.topbar-right { display:flex;align-items:center;gap:14px;flex-shrink:0; }
.server-indicator { display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:11px;letter-spacing:1px; }

/* ── Dots ────────────────────────────────────────────────── */
.dot { width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0; }
.dot-on    { background:var(--green);box-shadow:0 0 10px var(--green-d);animation:dot-p 2s ease-in-out infinite; }
.dot-off   { background:var(--text-dim); }
.dot-amber { background:var(--amber);box-shadow:0 0 10px rgba(255,170,0,.28);animation:dot-p .8s ease-in-out infinite; }
@keyframes dot-p { 0%,100%{opacity:1}50%{opacity:.3} }
.text-green { color:var(--green); }
.text-amber { color:var(--amber); }

/* ── Buttons ─────────────────────────────────────────────── */
.btn-icon {
  background:transparent;border:1px solid var(--border);border-radius:var(--r);
  color:var(--text);cursor:pointer;font-size:15px;
  width:30px;height:30px;display:flex;align-items:center;justify-content:center;
  transition:border-color .15s,color .15s;
}
.btn-icon:hover { border-color:var(--accent);color:var(--accent); }

.btn {
  background:transparent;border:1px solid var(--border);border-radius:var(--r);
  color:var(--text);cursor:pointer;
  font-family:var(--mono);font-size:11px;font-weight:700;letter-spacing:1px;
  padding:7px 12px;text-align:center;transition:all .15s;white-space:nowrap;
}
.btn:hover:not(:disabled) { border-color:var(--accent);color:var(--accent); }
.btn:disabled { opacity:.28;cursor:not-allowed; }

.btn-accent {
  border-color:var(--accent-d);color:var(--accent);background:rgba(0,212,255,.04);
}
.btn-accent:hover:not(:disabled) { background:rgba(0,212,255,.10);box-shadow:0 0 14px rgba(0,212,255,.16); }

.btn-green {
  border-color:rgba(0,232,122,.38);color:var(--green);background:rgba(0,232,122,.04);
}
.btn-green:hover:not(:disabled) { background:rgba(0,232,122,.10);box-shadow:0 0 10px rgba(0,232,122,.16); }

.btn-red {
  border-color:rgba(255,51,85,.34);color:var(--red);background:rgba(255,51,85,.04);
}
.btn-red:hover:not(:disabled) { background:rgba(255,51,85,.10); }

.btn-ghost { border-color:var(--border);color:var(--text); }
.btn-xs    { font-size:9px;padding:2px 6px;letter-spacing:0;flex-shrink:0; }
.full-width { width:100%;margin-top:6px; }

/* ── Layout ──────────────────────────────────────────────── */
.workspace { display:grid;grid-template-columns:280px 1fr;overflow:hidden; }

.sidebar {
  border-right:1px solid var(--border);
  overflow-y:auto;padding:10px;
  display:flex;flex-direction:column;gap:10px;
}

/* ── Panels ──────────────────────────────────────────────── */
.panel {
  background:var(--panel);border:1px solid var(--border);border-radius:6px;
  padding:13px;position:relative;overflow:hidden;
}
.panel::before {
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--accent-d),transparent);
}
.panel-header { display:flex;align-items:center;gap:8px;margin-bottom:13px; }
.panel-title  { font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:2px;color:var(--accent); }

/* step badge circles */
.step-badge {
  width:18px;height:18px;border-radius:50%;flex-shrink:0;
  border:1px solid var(--accent-d);color:var(--accent);
  font-family:var(--mono);font-size:10px;font-weight:700;
  display:flex;align-items:center;justify-content:center;
  text-shadow:0 0 6px var(--accent);
}

/* ── Form fields ─────────────────────────────────────────── */
.field-group { margin-bottom:9px; }
.field-label {
  display:block;font-family:var(--mono);font-size:9px;
  letter-spacing:1px;color:var(--text-dim);text-transform:uppercase;margin-bottom:4px;
}
.field-input,.field-select {
  width:100%;background:var(--panel-alt);border:1px solid var(--border);
  border-radius:var(--r);color:var(--text-hi);
  font-family:var(--mono);font-size:12px;padding:6px 8px;outline:none;
  transition:border-color .15s;appearance:none;-webkit-appearance:none;
}
.field-input:focus,.field-select:focus { border-color:var(--accent-d); }
.field-input-sm { max-width:100px; }
.field-select {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%2300d4ff'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 8px center;padding-right:24px;
}
.field-select option { background:#0c1121;color:#d4eeff; }

/* ── Model status badge ──────────────────────────────────── */
.model-status {
  min-height:30px;padding:6px 0;
  font-family:var(--mono);font-size:11px;
}
.badge-ready   { color:var(--green); }
.badge-missing { color:var(--amber); }
.badge-path {
  display:block;font-size:9px;color:var(--text-dim);
  margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}

/* ── Server controls ─────────────────────────────────────── */
.server-controls {
  display:grid;grid-template-columns:1fr 1fr;gap:6px;
}

.status-text {
  font-family:var(--mono);font-size:10px;color:var(--text-dim);
  margin:7px 0 4px;min-height:14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}

.endpoint-list { list-style:none;margin-top:6px;display:flex;flex-direction:column;gap:3px; }
.endpoint-list a {
  color:rgba(0,212,255,.45);font-family:var(--mono);font-size:9px;
  text-decoration:none;word-break:break-all;transition:color .15s;
}
.endpoint-list a:hover { color:var(--accent); }

/* ── Download progress bar ───────────────────────────────── */
.dl-progress-wrap {
  height:5px;background:var(--border-hi);border-radius:3px;margin-top:8px;overflow:hidden;
}
.dl-progress-fill {
  height:100%;background:var(--accent);border-radius:3px;
  width:0%;transition:width 1.2s linear;
  box-shadow:0 0 6px rgba(0,212,255,0.5);
}
.dl-progress-wrap.indeterminate .dl-progress-fill {
  width:40% !important;animation:dl-scan 1.4s ease-in-out infinite;
}
@keyframes dl-scan {
  0%   { transform:translateX(-120%); }
  100% { transform:translateX(300%); }
}

/* ── Downloaded quick-pick ───────────────────────────────── */
.dl-section-label {
  font-family:var(--mono);font-size:9px;letter-spacing:1px;
  color:var(--text-dim);text-transform:uppercase;margin:12px 0 5px;
}
.dl-scroll {
  max-height:120px;overflow-y:auto;
  scrollbar-width:thin;scrollbar-color:var(--border-hi) transparent;
}
.dl-scroll::-webkit-scrollbar { width:4px; }
.dl-scroll::-webkit-scrollbar-thumb { background:var(--border-hi);border-radius:2px; }
.dl-item {
  display:flex;align-items:center;gap:5px;
  padding:5px 6px;border-bottom:1px solid var(--border);
  border-radius:3px;transition:background .1s;
}
.dl-item-active {
  background:rgba(0,212,255,0.07);
  border-left:2px solid var(--accent);
  padding-left:4px;
}
.dl-item-active .dl-repo { color:var(--accent); }
.dl-item-active .dl-use  { border-color:var(--accent);color:var(--accent); }
.dl-check { color:var(--green);font-size:11px;flex-shrink:0; }
.dl-repo  { font-family:var(--mono);font-size:9px;color:var(--text);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }

/* ── Chat area ───────────────────────────────────────────── */
.chat-area { display:grid;grid-template-rows:44px 1fr auto;overflow:hidden; }
.chat-header {
  display:flex;align-items:center;gap:8px;padding:0 18px;
  border-bottom:1px solid var(--border);background:rgba(12,17,33,.7);
}
.chat-header-right { display:flex;align-items:center;gap:10px;margin-left:auto; }
.chat-conn { font-family:var(--mono);font-size:10px;letter-spacing:.5px;transition:color .3s; }
.chat-conn.online  { color:var(--green); }
.chat-conn.starting { color:var(--amber); }
.chat-conn.offline { color:var(--text-dim); }

/* ── Chat messages ───────────────────────────────────────── */
.chat-messages {
  overflow-y:auto;padding:20px 24px;
  display:flex;flex-direction:column;gap:22px;
}

.chat-empty {
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  flex:1;min-height:200px;gap:14px;color:var(--text-dim);user-select:none;
}
.chat-empty-logo {
  width:72px;height:72px;object-fit:contain;
  opacity:.25;filter:drop-shadow(0 0 16px rgba(0,180,255,0.4));
  animation:logo-float 4s ease-in-out infinite;
}
@keyframes logo-float { 0%,100%{transform:translateY(0);opacity:.25}50%{transform:translateY(-6px);opacity:.45} }

/* flow steps in empty state */
.flow-steps {
  list-style:none;display:flex;flex-direction:column;gap:6px;
  font-family:var(--mono);font-size:11px;letter-spacing:.5px;
}
.flow-steps li { display:flex;align-items:center;gap:8px; }
.flow-steps li::before {
  content:attr(data-n);
  width:16px;height:16px;border-radius:50%;
  border:1px solid var(--text-dim);
  display:flex;align-items:center;justify-content:center;
  font-size:9px;flex-shrink:0;
}
/* number each li via CSS counter */
.flow-steps { counter-reset:step; }
.flow-steps li { counter-increment:step; }
.flow-steps li::before { content:counter(step); }

.chat-msg { display:flex;gap:14px;animation:msg-in .18s ease-out; }
@keyframes msg-in { from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)} }

.msg-role {
  font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:1px;
  padding-top:3px;flex-shrink:0;width:28px;text-align:right;
}
.chat-msg-user      .msg-role { color:var(--accent); }
.chat-msg-assistant .msg-role { color:var(--green); }

.msg-body {
  font-size:14px;line-height:1.65;color:var(--text-hi);
  white-space:pre-wrap;word-break:break-word;flex:1;
}
.chat-msg-user .msg-body { color:var(--text); }

.msg-body code {
  font-family:var(--mono);font-size:12px;
  background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.14);
  border-radius:3px;padding:1px 5px;color:var(--accent);
}
.msg-body pre {
  background:rgba(0,0,0,.4);border:1px solid var(--border);
  border-radius:5px;padding:10px 12px;margin:8px 0;
  overflow-x:auto;font-family:var(--mono);font-size:12px;
  color:var(--text-hi);line-height:1.5;
}
.msg-body pre code { background:none;border:none;padding:0;color:inherit; }

.cursor { display:inline-block;color:var(--accent);animation:cur .75s step-end infinite; }
@keyframes cur { 0%,100%{opacity:1}50%{opacity:0} }

/* ── Chat input ──────────────────────────────────────────── */
.chat-input-area { border-top:1px solid var(--border);padding:12px 18px;background:var(--panel); }
.chat-input-row  { display:flex;align-items:flex-end;gap:9px;margin-bottom:5px; }
.prompt-arrow {
  font-family:var(--mono);font-size:20px;color:var(--accent);
  padding-bottom:7px;flex-shrink:0;text-shadow:0 0 8px var(--accent);
}
.chat-input {
  flex:1;background:var(--panel-alt);border:1px solid var(--border);
  border-radius:var(--r);color:var(--text-hi);font-family:var(--sans);
  font-size:14px;line-height:1.5;padding:8px 11px;resize:none;outline:none;
  transition:border-color .15s;min-height:38px;max-height:140px;overflow-y:auto;
}
.chat-input:focus { border-color:var(--accent-d); }
.chat-input::placeholder { color:var(--text-dim); }
.chat-hints {
  display:flex;justify-content:space-between;
  font-family:var(--mono);font-size:9px;color:var(--text-dim);letter-spacing:.5px;
}

/* ── Perf stats line ─────────────────────────────────────── */
.msg-stats {
  font-family:var(--mono);font-size:10px;color:var(--accent-d);
  margin-top:8px;letter-spacing:0.5px;border-top:1px solid var(--border);padding-top:5px;
}

/* ── Mobile ──────────────────────────────────────────────── */
@media (max-width:760px) {
  .workspace { grid-template-columns:1fr;grid-template-rows:auto 1fr; }
  .sidebar   { border-right:none;border-bottom:1px solid var(--border);max-height:45vh; }
  body       { overflow:auto; }
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

function shortenPath(p) {
  if (!p) return '';
  const parts = p.split('/');
  return parts.length > 4 ? '…/' + parts.slice(-3).join('/') : p;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function selectedCatalogEntry() {
  return app.catalog.find(e => e.repo_id === $('model').value) || null;
}

function clearManualModelDirOrigin() {
  delete $('model-dir').dataset.repoId;
}

// ── App state ─────────────────────────────────────────────────────────────────

const app = {
  catalog:          [],
  serverRunning:    false,
  serverStarting:   false,
  serverPort:       8080,
  serverModelDir:   null,
  serverModelId:    null,
  downloadedModels: [],
  streaming:        false,
};

// ── State → UI ────────────────────────────────────────────────────────────────

function applyState(data) {
  app.catalog          = data.catalog || [];
  app.serverRunning    = !!(data.server && data.server.running);
  app.serverStarting   = !!(data.server && data.server.starting);
  app.serverPort       = (data.server && data.server.port) || 8080;
  app.downloadedModels = data.downloaded_models || [];

  // Clear chat when the active model changes, including after a stop+restart cycle.
  // app.serverModelDir tracks the last non-null model_dir; we only update it when
  // a server is running so a stop→restart transition triggers a clear correctly.
  const newModelDir = (data.server && data.server.model_dir) || null;
  if (newModelDir && newModelDir !== app.serverModelDir) {
    clearChat();
  }
  if (newModelDir) app.serverModelDir = newModelDir;
  app.serverModelId = (data.server && data.server.model_id) || null;

  $('sys-status').textContent = data.status || '';

  // Top-bar server dot
  const dot = $('server-dot'), lbl = $('server-label');
  if (app.serverRunning) {
    dot.className = 'dot dot-on'; lbl.textContent = 'ONLINE'; lbl.className = 'text-green';
  } else if (app.serverStarting) {
    dot.className = 'dot dot-amber'; lbl.textContent = 'STARTING'; lbl.className = 'text-amber';
  } else {
    dot.className = 'dot dot-off'; lbl.textContent = 'OFFLINE'; lbl.className = '';
  }

  $('server-status').textContent = (data.server && data.server.status) || '';
  renderEndpoints((data.server && data.server.endpoints) || []);
  if (data.server && data.server.engine) {
    $('engine').value = data.server.engine;
  }

  fillModelSelectors(data);
  renderDownloaded(app.downloadedModels);
  updateModelStatus();  // drives model-dir auto-fill + download button label

  // Chat badge
  const conn = $('chat-conn');
  if (app.serverRunning) {
    conn.textContent = `→ 127.0.0.1:${app.serverPort}`;
    conn.className   = 'chat-conn online';
  } else if (app.serverStarting) {
    conn.textContent = 'server starting';
    conn.className   = 'chat-conn starting';
  } else {
    conn.textContent = 'server offline';
    conn.className   = 'chat-conn offline';
  }

  $('chat-model-label').textContent =
    data.server && data.server.model_id ? `model: ${shortModelName(data.server.model_id)}` : '';
}

// ── Model selectors ───────────────────────────────────────────────────────────

function kinds()           { return [...new Set(app.catalog.map(e => e.kind))]; }
function familiesOf(kind)  { return [...new Set(app.catalog.filter(e => e.kind === kind).map(e => e.family))]; }
function modelsOf(k, fam)  { return app.catalog.filter(e => e.kind === k && e.family === fam); }

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
    `<option value="${m.repo_id}">${m.label}</option>`
  ).join('');
  if (preferred && models.some(m => m.repo_id === preferred)) {
    $('model').value = preferred;
  }
  updateModelStatus();
}

// ── Model status: the key link between model selection and server ─────────────

function updateModelStatus() {
  const repo       = $('model').value;
  const downloaded = app.downloadedModels.find(m => m.repo_id === repo);
  const statusEl   = $('model-status');
  const dlBtn      = $('download');

  if (downloaded && downloaded.path) {
    // Model is available locally
    statusEl.innerHTML =
      `<span class="badge-ready">● READY</span>` +
      `<span class="badge-path">${escapeHtml(shortenPath(downloaded.path))}</span>`;

    // Auto-fill model-dir for the server (user can still override)
    if (!$('model-dir').dataset.userEdited) {
      $('model-dir').value = downloaded.path;
    }

    dlBtn.textContent  = 'RE-DOWNLOAD';
    dlBtn.className    = 'btn btn-ghost full-width';
  } else {
    // Not downloaded yet
    statusEl.innerHTML = `<span class="badge-missing">○ NOT DOWNLOADED</span>`;

    // Clear model-dir if it was auto-filled from a different downloaded model
    if (!$('model-dir').dataset.userEdited) {
      $('model-dir').value = '';
    }

    dlBtn.textContent = 'DOWNLOAD';
    dlBtn.className   = 'btn btn-accent full-width';
  }
}

// ── Endpoints ─────────────────────────────────────────────────────────────────

function renderEndpoints(eps) {
  $('endpoint-list').innerHTML = eps.slice(0, 4).map(ep =>
    `<li><a href="${escapeHtml(ep.url)}" target="_blank" rel="noopener">${escapeHtml(ep.label)}</a></li>`
  ).join('');
}

// ── Downloaded quick-pick ─────────────────────────────────────────────────────

function shortModelName(repoId) {
  // "mlx-community/Qwen3-4B-4bit" → "Qwen3-4B-4bit"
  return repoId.split('/').pop() || repoId;
}

function renderDownloaded(models) {
  const el = $('downloaded-list');
  if (!models.length) { el.innerHTML = ''; return; }

  const activeRepo = $('model').value;

  el.innerHTML =
    `<div class="dl-section-label">DOWNLOADED</div>` +
    `<div class="dl-scroll">` +
    models.map(m => {
      const active = m.repo_id === activeRepo ? ' dl-item-active' : '';
      const name   = shortModelName(m.repo_id);
      const manual = app.catalog.some(e => e.repo_id === m.repo_id) ? '0' : '1';
      const repo   = escapeHtml(m.repo_id);
      const path   = escapeHtml(m.path || '');
      return `
      <div class="dl-item${active}" data-repo="${repo}">
        <span class="dl-check">✓</span>
        <span class="dl-repo" title="${path}">${escapeHtml(name)}</span>
        <button class="btn btn-ghost btn-xs dl-use"
          data-path="${path}"
          data-repo="${repo}"
          data-manual="${manual}">USE</button>
      </div>`;
    }).join('') +
    `</div>`;

  el.querySelectorAll('.dl-use').forEach(btn => {
    btn.addEventListener('click', () => {
      const repo = btn.dataset.repo;
      const entry = app.catalog.find(e => e.repo_id === repo);
      const manual = btn.dataset.manual === '1' || !entry;
      if (entry) {
        delete $('model-dir').dataset.userEdited;
        clearManualModelDirOrigin();
        $('model-kind').value = entry.kind;
        $('family').innerHTML = familiesOf(entry.kind).map(f => `<option value="${f}">${f}</option>`).join('');
        $('family').value = entry.family;
        fillSizes(entry.kind, entry.family, repo);
      }
      $('model-dir').value = btn.dataset.path;
      if (manual) {
        $('model-dir').dataset.userEdited = '1';
        $('model-dir').dataset.repoId = repo || 'local';
      } else {
        delete $('model-dir').dataset.userEdited;
        clearManualModelDirOrigin();
      }

      // Highlight this row, remove highlight from others
      el.querySelectorAll('.dl-item').forEach(row => row.classList.remove('dl-item-active'));
      btn.closest('.dl-item').classList.add('dl-item-active');

      updateModelStatus();
    });
  });
}

// ── Server controls ───────────────────────────────────────────────────────────

function serverPayload() {
  const entry = selectedCatalogEntry();
  const manual = $('model-dir').dataset.userEdited === '1';
  const manualRepo = $('model-dir').dataset.repoId || '';
  return JSON.stringify({
    port:      Number($('port').value || 8080),
    repo_id:   manual && manualRepo ? manualRepo : (entry ? entry.repo_id : ''),
    model_dir: $('model-dir').value.trim(),
    manual_model_dir: manual,
    engine:    $('engine').value,
  });
}

function setBusy(busy) {
  ['start-server', 'stop-server', 'restart-server'].forEach(id => $(id).disabled = busy);
}

async function startServer() {
  const modelDir = $('model-dir').value.trim();
  if (!modelDir) {
    $('server-status').textContent = 'Select and download a model first.';
    return;
  }
  setBusy(true);
  $('server-dot').className   = 'dot dot-amber';
  $('server-status').textContent = 'Starting…';
  try {
    await api('/api/server/start', { method: 'POST', body: serverPayload() });
  } catch (err) {
    $('server-status').textContent = err.message;
  }
  await loadState();
  setBusy(false);
}

async function stopServer() {
  setBusy(true);
  try {
    await api('/api/server/stop', { method: 'POST', body: '{}' });
  } catch (err) {
    $('server-status').textContent = err.message;
  }
  await loadState();
  setBusy(false);
}

async function restartServer() {
  const modelDir = $('model-dir').value.trim();
  if (!modelDir) {
    $('server-status').textContent = 'No model path configured.';
    return;
  }
  setBusy(true);
  $('server-dot').className   = 'dot dot-amber';
  $('server-status').textContent = 'Restarting…';
  try {
    await api('/api/server/restart', { method: 'POST', body: serverPayload() });
  } catch (err) {
    $('server-status').textContent = err.message;
  }
  await loadState();
  setBusy(false);
}

// ── Download ──────────────────────────────────────────────────────────────────

async function startDownload() {
  const entry = app.catalog.find(e => e.repo_id === $('model').value);
  if (!entry) return;

  $('download').disabled = true;
  $('download-status').textContent = `Downloading ${entry.repo_id}…`;
  $('dl-progress-wrap').hidden = false;
  $('dl-progress-wrap').classList.add('indeterminate');
  $('dl-progress-fill').style.width = '0%';

  try {
    const job = await api('/api/download', {
      method: 'POST',
      body: JSON.stringify({ repo_id: entry.repo_id, kind: entry.kind }),
    });
    pollDownload(job.id, entry.repo_id);
  } catch (err) {
    $('dl-progress-wrap').hidden = true;
    $('download-status').textContent = err.message;
    $('download').disabled = false;
  }
}

async function pollDownload(jobId, repoId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    await loadState(); // refreshes downloadedModels → updateModelStatus auto-fills model-dir
    if (job.status === 'running') {
      const pct = Math.min(job.progress || 0, 99);
      if (pct > 0) {
        $('dl-progress-wrap').classList.remove('indeterminate');
        $('dl-progress-fill').style.width = pct + '%';
      }
      $('download-status').textContent = job.message || `Downloading ${repoId}…`;
      setTimeout(() => pollDownload(jobId, repoId), 1500);
    } else {
      $('dl-progress-wrap').classList.remove('indeterminate');
      $('dl-progress-fill').style.width = job.status === 'succeeded' ? '100%' : $('dl-progress-fill').style.width;
      setTimeout(() => { $('dl-progress-wrap').hidden = true; }, 800);
      $('download').disabled = false;
      $('download-status').textContent =
        job.status === 'succeeded' ? `✓ ${repoId} ready — start the server` : `✗ ${job.message || 'download failed'}`;
    }
  } catch (err) {
    $('dl-progress-wrap').classList.remove('indeterminate');
    $('dl-progress-wrap').hidden = true;
    $('download').disabled = false;
    $('download-status').textContent = err.message;
  }
}

// ── Chat ──────────────────────────────────────────────────────────────────────

const MAX_CHAT_HISTORY_TURNS = 20;
const MAX_CHAT_HISTORY_MESSAGES = MAX_CHAT_HISTORY_TURNS * 2;
const history = [];

function ensureChatOpen() {
  const empty = $('chat-empty');
  if (empty) empty.remove();
}

function renderMd(text) {
  const esc = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  return esc
    .replace(/```[\w]*\n([\s\S]*?)```/g, (_,c) => `<pre><code>${c.trimEnd()}</code></pre>`)
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

function addMsgEl(role, content, streaming) {
  ensureChatOpen();
  const wrap = document.createElement('div');
  wrap.className = `chat-msg chat-msg-${role}`;
  const lbl  = document.createElement('span');
  lbl.className = 'msg-role';
  lbl.textContent = role === 'user' ? 'YOU' : 'AX';
  const body = document.createElement('div');
  body.className = 'msg-body';
  if (streaming) {
    body.appendChild(makeCursor());
  } else if (role === 'assistant') {
    body.innerHTML = renderMd(content);
  } else {
    body.textContent = content;
  }
  wrap.append(lbl, body);
  $('chat-messages').append(wrap);
  scrollChat();
  return body;
}

function makeCursor() {
  const s = document.createElement('span');
  s.className = 'cursor'; s.textContent = '▊';
  return s;
}

function appendToken(bodyEl, token) {
  const cur = bodyEl.querySelector('.cursor');
  if (cur) cur.remove();
  bodyEl.insertAdjacentText('beforeend', token);
  bodyEl.appendChild(makeCursor());
  scrollChat();
}

function finalizeBody(bodyEl, full) {
  const cur = bodyEl.querySelector('.cursor');
  if (cur) cur.remove();
  bodyEl.innerHTML = renderMd(full);
  scrollChat();
}

function scrollChat() {
  const el = $('chat-messages');
  el.scrollTop = el.scrollHeight;
}

function recentChatHistory() {
  const recent = history.slice(-MAX_CHAT_HISTORY_MESSAGES);
  while (recent.length && recent[0].role !== 'user') recent.shift();
  return recent;
}

function trimChatHistory() {
  while (history.length > MAX_CHAT_HISTORY_MESSAGES) history.shift();
  while (history.length && history[0].role !== 'user') history.shift();
}

async function sendMessage() {
  const input = $('chat-input');
  const text  = input.value.trim();
  if (!text || app.streaming) return;

  if (app.serverStarting) {
    $('server-status').textContent = 'Server is still starting. Try again when it is online.';
    return;
  }
  if (!app.serverRunning) {
    $('server-status').textContent = 'Start the server first (Step 2).';
    return;
  }

  input.value = ''; autoResize(input);
  history.push({ role: 'user', content: text });
  addMsgEl('user', text, false);

  const bodyEl = addMsgEl('assistant', '', true);
  app.streaming = true;
  $('chat-send').disabled = true;
  $('chat-send').textContent = '···';

  let accumulated = '';
  let streamError = false;
  let perfStats = null;

  try {
    const msgsToSend = [
      { role: 'system', content: 'You are a helpful assistant.' },
      ...recentChatHistory(),
    ];
    const res = await fetch('/api/proxy/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ messages: msgsToSend, stream: true, max_tokens: 2048 }),
    });

    if (!res.ok) {
      let msg;
      try {
        const e = (await res.json()).error;
        if (typeof e === 'string') msg = e;
        else if (e && typeof e.message === 'string') msg = e.message;
        else msg = JSON.stringify(e);
      } catch { msg = res.statusText; }
      throw new Error(msg || `HTTP ${res.status}`);
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let sseBuf = '';
    let sawDone = false;

    while (!sawDone) {
      const { done, value } = await reader.read();
      if (done) break;
      sseBuf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = sseBuf.indexOf('\n')) !== -1) {
        const line = sseBuf.slice(0, nl).trim();
        sseBuf = sseBuf.slice(nl + 1);
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') { sawDone = true; break; }
        let chunk;
        try { chunk = JSON.parse(payload); } catch { continue; }
        const delta = (chunk.choices?.[0]?.delta?.content) || '';
        if (delta) { accumulated += delta; appendToken(bodyEl, delta); }
        if (chunk.usage && chunk.choices?.[0]?.finish_reason) {
          perfStats = chunk.usage;
        }
      }
    }
  } catch (err) {
    accumulated = `[error: ${err.message}]`;
    streamError = true;
  }

  finalizeBody(bodyEl, accumulated || '(empty response)');
  if (perfStats && perfStats.generation_tps > 0) {
    const statsEl = document.createElement('div');
    statsEl.className = 'msg-stats';
    const prompt = perfStats.prompt_tokens || 0;
    const output = perfStats.completion_tokens || 0;
    const tps = Number(perfStats.generation_tps).toFixed(1);
    const engineLabel = $('engine').options[$('engine').selectedIndex]?.text || '';
    const modelLabel = app.serverModelId ? shortModelName(app.serverModelId) : '';
    const prefix = [engineLabel, modelLabel].filter(Boolean).join(' · ');
    statsEl.textContent = `⚡ ${prefix} · ${prompt} prompt · ${output} output · ${tps} tok/s`;
    bodyEl.appendChild(statsEl);
  }
  // Only add real content to history — empty or error responses would appear
  // as the assistant's previous turn and confuse the model on the next request.
  if (accumulated && !streamError) {
    history.push({ role: 'assistant', content: accumulated });
    trimChatHistory();
  }
  app.streaming = false;
  $('chat-send').disabled = false;
  $('chat-send').textContent = 'SEND';
}

function clearChat() {
  history.length = 0;
  const msgs = $('chat-messages');
  msgs.innerHTML = '';
  const empty = document.createElement('div');
  empty.className = 'chat-empty'; empty.id = 'chat-empty';
  const logoSrc = document.querySelector('.brand-logo')?.src || '';
  empty.innerHTML =
    '<img src="' + logoSrc + '" class="chat-empty-logo" alt="AX Engine">' +
    '<ol class="flow-steps"><li>Select a model above</li><li>Download it if needed</li><li>Start the server</li><li>Chat here</li></ol>';
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
    // While server is starting, keep polling until it's running or stopped.
    const starting = !!(data.server && data.server.starting);
    if (starting) {
      scheduleServerPoll();
    } else {
      cancelServerPoll();
    }
  } catch (err) {
    $('sys-status').textContent = err.message;
    // Keep polling if the server was in "starting" state — a transient request
    // error shouldn't leave the UI stuck with no way to detect readiness.
    if (app.serverStarting) {
      scheduleServerPoll();
    }
  }
}

let _serverPollTimer = null;
function scheduleServerPoll() {
  if (_serverPollTimer) return;
  _serverPollTimer = setTimeout(async () => {
    _serverPollTimer = null;
    await loadState();
  }, 2000);
}
function cancelServerPoll() {
  if (_serverPollTimer) { clearTimeout(_serverPollTimer); _serverPollTimer = null; }
}

// ── Wire events ───────────────────────────────────────────────────────────────

$('model-kind').addEventListener('change', () => {
  // User explicitly changed the model — let model-dir follow the new selection.
  delete $('model-dir').dataset.userEdited;
  clearManualModelDirOrigin();
  const kind = $('model-kind').value;
  const fams = familiesOf(kind);
  $('family').innerHTML = fams.map(f => `<option value="${f}">${f}</option>`).join('');
  fillSizes(kind, fams[0], null);
});

$('family').addEventListener('change', () => {
  delete $('model-dir').dataset.userEdited;
  clearManualModelDirOrigin();
  fillSizes($('model-kind').value, $('family').value, null);
});

$('model').addEventListener('change', () => {
  delete $('model-dir').dataset.userEdited;
  clearManualModelDirOrigin();
  updateModelStatus();
});

// Mark model-dir as user-edited when typed manually
$('model-dir').addEventListener('input', () => {
  $('model-dir').dataset.userEdited = '1';
  clearManualModelDirOrigin();
  if (!$('model-dir').value) delete $('model-dir').dataset.userEdited;
});

$('download').addEventListener('click', startDownload);
$('start-server').addEventListener('click', startServer);
$('stop-server').addEventListener('click', stopServer);
$('restart-server').addEventListener('click', restartServer);
$('refresh').addEventListener('click', loadState);
$('chat-clear').addEventListener('click', clearChat);
$('chat-send').addEventListener('click', sendMessage);

$('chat-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
$('chat-input').addEventListener('input', function() { autoResize(this); });

// ── Boot ──────────────────────────────────────────────────────────────────────

loadState().catch(err => { $('sys-status').textContent = String(err.message || err); });
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
        LoadState::Ready(_) => "ready",
        LoadState::Unavailable(_) => "unavailable",
        LoadState::NotLoaded(_) => "not_loaded",
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
    fn manager_js_preserves_local_quick_pick_as_manual_model_dir() {
        let js = manager_js();

        assert!(js.contains("data-manual"));
        assert!(js.contains("dataset.repoId"));
        assert!(js.contains("manual && manualRepo ? manualRepo"));
    }

    #[test]
    fn manager_js_escapes_downloaded_model_markup_values() {
        let js = manager_js();

        assert!(js.contains("function escapeHtml"));
        assert!(js.contains("const repo   = escapeHtml(m.repo_id);"));
        assert!(js.contains("const path   = escapeHtml(m.path || '');"));
        assert!(js.contains("${escapeHtml(shortenPath(downloaded.path))}"));
        assert!(js.contains("href=\"${escapeHtml(ep.url)}\""));
        assert!(js.contains("${escapeHtml(ep.label)}"));
        assert!(!js.contains("data-path=\"${m.path || ''}\""));
        assert!(!js.contains("href=\"${ep.url}\""));
    }

    #[test]
    fn manager_js_bounds_chat_history_sent_to_proxy() {
        let js = manager_js();

        assert!(js.contains("const MAX_CHAT_HISTORY_TURNS = 20;"));
        assert!(js.contains("function recentChatHistory()"));
        assert!(js.contains("...recentChatHistory(),"));
        assert!(js.contains("trimChatHistory();"));
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
