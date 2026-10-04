// Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Inference engines in the launch form (core/engines on the server).
//
// A model's engine comes from its library entry: files are llama.cpp, the
// virtual strata/... entries are Strata. Elements tagged
// data-engine-only="<engine>" are shown only for that engine, so the
// llama.cpp-only fields disappear for Strata and the Strata section appears.
// The server enforces the same rules (rejects llama.cpp-only fields, forces
// Max Concurrent to 1); this file only keeps the form honest.
// -------------------------------------------------------------------------

let launchEngine = 'llamacpp';
let launchSelectedModel = null;    // library entry of the selected file
let launchEngineModel = null;      // its entry in the engine's catalogue (Strata)
let _enginesCache = {};            // nodeId -> {at, data} from /api/engines
const _ENGINES_TTL_MS = 15000;

function currentLaunchEngine() { return launchEngine; }

function engineForModel(m) {
  return 'llamacpp';
}

// The Inference Engine dropdown: Strata only for a file Strata can run
// (engine_models.strata, a downloaded recommended model) on a node that can
// run Strata.
async function updateEngineSelect() {
  updateStrataModelsButton();
  const sel = document.getElementById('f-engine');
  if (!sel) return;
  sel.value = launchEngine;
  const opt = sel.querySelector('option[value="strata"]');
  if (!opt) return;
  const m = launchSelectedModel;
  let reason = '';
  if (!m) reason = 'Select a model first';
  else if (!(m.engine_models && m.engine_models.strata)) reason = 'Strata runs only its recommended models (download one from the Strata settings)';
  else {
    const info = _engineInfo(await fetchEnginesForNode(_launchNode()), 'strata');
    if (!info) reason = 'Strata is unknown on the target node';
    else if (!info.available) reason = info.reason || 'Strata is unavailable on the target node';
  }
  // Keep the current engine selectable even if it became unavailable, so the
  // form never shows a value it can't hold; the server re-checks on launch.
  opt.disabled = !!reason && launchEngine !== 'strata';
  opt.title = reason;
  sel.title = reason && launchEngine !== 'strata' ? reason : '';
}

const engineSelectEl = document.getElementById('f-engine');
if (engineSelectEl) engineSelectEl.addEventListener('change', () => {
  applyEngineToLaunchForm(engineSelectEl.value, launchSelectedModel);
});

async function fetchEnginesForNode(nodeId, force = false) {
  const key = nodeId || 'local';
  const hit = _enginesCache[key];
  if (!force && hit && Date.now() - hit.at < _ENGINES_TTL_MS) return hit.data;
  try {
    const res = await _nf(nodeId, '/api/engines');
    if (!res || !res.ok) return hit ? hit.data : null;
    const data = await res.json();
    _enginesCache[key] = { at: Date.now(), data };
    return data;
  } catch (e) {
    return hit ? hit.data : null;
  }
}

function _engineInfo(data, name) {
  return ((data && data.engines) || []).find(e => e.name === name) || null;
}

// Virtual models a peer can launch, from its heartbeat snapshot
// (snapshot.system.engines, see api/cluster.build_local_snapshot).
function engineModelsFromSnapshot(node) {
  const engines = (((node || {}).snapshot || {}).system || {}).engines;
  const out = [];
  ((engines && engines.engines) || []).forEach(e => {
    if (e.available && Array.isArray(e.models)) out.push(...e.models);
  });
  return out;
}

// Instance card helpers (used by renderInstances).
function instanceEngineBadge(inst) {
  if (!inst || !inst.engine || inst.engine === 'llamacpp') return '';
  const label = inst.engine === 'strata' ? 'Strata' : inst.engine;
  return ` <span class="badge badge-engine" title="Inference engine">${escHtml(label)}</span>`;
}

function instanceServerLabel(inst) {
  return (inst && inst.engine === 'strata') ? 'Strata' : 'llama-server';
}

function instanceLoadStageLine(inst) {
  const st = inst && inst.status === 'starting' ? inst.load_stage : null;
  if (!st) return '';
  const pct = (st.percent != null) ? Math.max(0, Math.min(100, st.percent)) : null;
  const bar = pct != null
    ? `<div class="load-stage-bar"><div style="width:${pct}%"></div></div>
       <span class="load-stage-pct">${pct}%</span>`
    : '';
  const failed = st.stage === 'setup failed';
  return `<div class="meta load-stage-line${failed ? ' text-danger' : ''}">
    <span class="load-stage-name">${escHtml(st.stage)}</span>${bar}
    ${st.detail ? `<span class="load-stage-detail">${escHtml(st.detail)}</span>` : ''}
  </div>`;
}

// -------------------------------------------------------------------------
// Launch form
// -------------------------------------------------------------------------

function applyEngineToLaunchForm(engine, model) {
  launchEngine = engine || 'llamacpp';
  if (model !== undefined) launchSelectedModel = model || null;
  launchEngineModel = null;          // resolved from the catalogue in refreshStrataSection
  updateEngineSelect();
  document.querySelectorAll('[data-engine-only]').forEach(el => {
    el.hidden = el.dataset.engineOnly !== launchEngine;
  });

  const mc = document.getElementById('f-max-concurrent');
  const ctx = document.getElementById('f-ctx-size');
  if (launchEngine === 'strata') {
    // Strata serves one request at a time; the server forces the gate to 1
    // and queues the rest in llamaman.
    if (mc) { mc.value = 1; mc.disabled = true; mc.title = 'Strata serves one request at a time; extra requests queue in llamaMan'; }
    if (ctx) ctx.setAttribute('list', 'strata-context-options');
    refreshStrataSection();
  } else {
    if (mc) { mc.disabled = false; mc.title = ''; }
    if (ctx) ctx.removeAttribute('list');
  }
  if (typeof populateLaunchImageSelect === 'function') populateLaunchImageSelect();
  // Sections shared by both engines whose state depends on which one it is.
  if (typeof updateMmprojState === 'function') updateMmprojState();
  if (typeof updateGpuSettingsState === 'function') updateGpuSettingsState();
  if (typeof updateModelSettingsState === 'function') updateModelSettingsState();
}

const strataVisionSelect = document.getElementById('f-strata-vision');
if (strataVisionSelect) strataVisionSelect.addEventListener('change', () => {
  if (typeof updateMmprojState === 'function') updateMmprojState();
});

// The launch form's image dropdown for a non-llama.cpp engine: just that
// engine's image (Strata's is built locally, never pulled).
async function populateEngineImageSelect() {
  const sel = document.getElementById('f-image');
  if (!sel) return;
  const data = await fetchEnginesForNode(_launchNode());
  const info = _engineInfo(data, launchEngine);
  sel.innerHTML = '';
  const name = (info && info.image) || '';
  if (!name) return;
  let present = null;
  try {
    const res = await _nf(_launchNode(), '/api/images');
    if (res && res.ok) {
      const imgs = (await res.json()).engine_images || [];
      const rec = imgs.find(i => i.name === name);
      if (rec) present = !!rec.present;
    }
  } catch (e) { /* ignore */ }
  const opt = document.createElement('option');
  opt.value = name;
  opt.textContent = name + (present === false ? '  (not built)' : '');
  sel.appendChild(opt);
  sel.value = name;
}

function _parseMemoryLimitGb(text) {
  const m = /^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)b?\s*$/i.exec(text || '');
  if (!m) return null;
  const n = parseFloat(m[1]);
  const unit = (m[2] || '').toLowerCase();
  const div = { '': 1024 ** 3, k: 1024 ** 2, m: 1024, g: 1, t: 1 / 1024 }[unit];
  return n / div;
}

async function refreshStrataSection() {
  if (launchEngine !== 'strata') return;
  const data = await fetchEnginesForNode(_launchNode());
  const info = _engineInfo(data, 'strata') || {};
  const modelId = launchSelectedModel && launchSelectedModel.engine_models
    ? launchSelectedModel.engine_models.strata : null;
  launchEngineModel = (info.models || []).find(x => x.name === modelId) || null;
  const m = launchEngineModel || {};

  const summary = document.getElementById('strata-model-summary');
  if (summary) {
    const parts = [];
    if (m.title) parts.push(m.title);
    if (m.size_display) parts.push(`${m.size_display} download`);
    if (m.ram_gb) parts.push(`needs ~${m.ram_gb} GB RAM`);
    if (m.experimental) parts.push('experimental');
    summary.textContent = parts.join(' · ');
  }

  const dl = document.getElementById('strata-context-options');
  if (dl && dl.childElementCount === 0) {
    (info.context_choices || []).forEach(v => {
      const opt = document.createElement('option');
      opt.value = v;
      dl.appendChild(opt);
    });
  }
  const ctx = document.getElementById('f-ctx-size');
  if (ctx && !ctx.value) ctx.value = 32768;

  const vision = document.getElementById('f-strata-vision');
  if (vision) {
    const noVision = m.vision === false;
    [...vision.options].forEach(o => { if (o.value !== 'no') o.disabled = noVision; });
    if (noVision) vision.value = 'no';
  }

  updateStrataWarnings();
}


// Strata's caveats, each under the Container & Proxy field it is about.
function updateStrataWarnings() {
  const show = (id, text) => {
    const ul = document.getElementById(id);
    if (ul) ul.innerHTML = (launchEngine === 'strata' && text)
      ? `<li><i class="fa-solid fa-triangle-exclamation"></i> ${escHtml(text)}</li>` : '';
  };
  const memText = document.getElementById('f-memory-limit')?.value.trim();
  const memGb = memText ? _parseMemoryLimitGb(memText) : null;
  show('strata-memory-warning', !memText ? ''
    : (memGb != null && memGb < 64)
      ? `Below ~64 GB: Strata loads 32-62 GB and may be OOM-killed. A memory limit also forces Low-RAM mode on.`
      : 'A memory limit forces Strata\'s Low-RAM mode on (its setup cannot see container limits).');
  const idle = parseInt(document.getElementById('f-idle-timeout')?.value, 10) || 0;
  show('strata-idle-warning', idle > 0
    ? 'A cold start loads 32-62 GB and takes minutes; the request that wakes it waits that long.' : '');
}

function applyStrataPresetToLaunchForm(p) {
  if (!p) return;
  const v = document.getElementById('f-strata-vision');
  if (v) v.value = ['no', 'yes', 'cpu'].includes(p.strata_vision) ? p.strata_vision : 'no';
  const kv = document.getElementById('f-strata-kv');
  if (kv) kv.value = ['', 'int8', 'q4_0', 'k8v4'].includes(p.strata_kv || '') ? (p.strata_kv || '') : '';
  const lr = document.getElementById('f-strata-low-ram');
  if (lr) lr.value = ['auto', 'on', 'off'].includes(p.strata_low_ram) ? p.strata_low_ram : 'auto';
  const ls = document.getElementById('f-strata-layer-split');
  if (ls) ls.value = p.strata_layer_split || '';
  if (typeof updateMmprojState === 'function') updateMmprojState();
  if (launchEngine === 'strata') {
    const mc = document.getElementById('f-max-concurrent');
    if (mc) mc.value = 1;
  }
}

// The launch / preset body for a Strata model: only the fields Strata reads
// (core/engines/strata.py launch_fields). The hidden llama.cpp fields are
// never sent, so a value left over from another model can't trip the
// server's "does not support" check.
function readStrataLaunchForm() {
  const ctxSizeRaw = document.getElementById('f-ctx-size').value.trim();
  if (!ctxSizeRaw) throw new Error('Context size is required');
  const ctxSize = parseInt(ctxSizeRaw, 10);
  if (!Number.isInteger(ctxSize) || ctxSize <= 0) throw new Error('Context size must be a positive integer');
  const val = id => document.getElementById(id);
  const body = {
    engine: 'strata',
    ctx_size: ctxSize,
    gpu_devices: val('f-gpu-devices').value.trim(),
    idle_timeout_min: parseInt(val('f-idle-timeout').value) || 0,
    max_concurrent: 1,
    max_queue_depth: parseInt(val('f-max-queue-depth').value) || 200,
    share_queue: val('f-share-queue').checked,
    share_queue_group: val('f-share-queue-group')?.value.trim() || '',
    share_queue_fallback: val('f-share-queue-fallback')?.checked || false,
    auto_restart_on_crash: val('f-auto-restart').checked,
    proxy_sampling_override_enabled: val('f-proxy-sampling-override-enabled').checked,
    proxy_sampling_temperature: parseFloat(val('f-proxy-sampling-temperature').value),
    proxy_sampling_top_k: parseInt(val('f-proxy-sampling-top-k').value, 10),
    proxy_sampling_top_p: parseFloat(val('f-proxy-sampling-top-p').value),
    proxy_sampling_presence_penalty: parseFloat(val('f-proxy-sampling-presence-penalty').value),
    proxy_sampling_repeat_penalty: parseFloat(val('f-proxy-sampling-repeat-penalty').value),
    loop_detect_enabled: val('f-loop-detect-enabled')?.checked || false,
    loop_detect_min_chunk_chars: parseInt(val('f-loop-detect-min-chunk-chars')?.value, 10) || 200,
    loop_detect_min_repetitions: parseInt(val('f-loop-detect-min-repetitions')?.value, 10) || 3,
    loop_detect_max_buffer_chars: parseInt(val('f-loop-detect-max-buffer-chars')?.value, 10) || 8192,
    loop_detect_scan_every_n_tokens: parseInt(val('f-loop-detect-scan-every-n-tokens')?.value, 10) || 64,
    loop_detect_scan_interval_s: parseInt(val('f-loop-detect-scan-interval-s')?.value, 10) || 10,
    strata_vision: val('f-strata-vision').value,
    strata_kv: val('f-strata-kv').value,
    strata_low_ram: val('f-strata-low-ram').value,
    strata_layer_split: val('f-strata-layer-split')?.value.trim() || '',
    pdf_input_enabled: val('f-pdf-input-enabled')?.checked || false,
    pdf_extract_text_first: val('f-pdf-extract-text-first')?.checked || false,
    pdf_dpi: parseInt(val('f-pdf-dpi')?.value, 10) || 200,
    pdf_max_pages: parseInt(val('f-pdf-max-pages')?.value, 10) || 20,
  };
  if (typeof validateProxySamplingBody === 'function') validateProxySamplingBody(body);
  const memoryLimit = val('f-memory-limit').value.trim();
  if (memoryLimit) body.memory_limit = memoryLimit;
  const image = val('f-image')?.value;
  if (image) body.image = image;
  return body;
}

// -------------------------------------------------------------------------
// Strata models modal: the catalogue, each downloadable into the target
// node's models folder (POST /api/engines/strata/download). Downloaded
// models appear in the library as ordinary files.
// -------------------------------------------------------------------------

function _strataModelStatus(m) {
  if (m.local_shards) return '<span class="badge badge-ok">downloaded</span>';
  const d = m.download;
  if (d && (d.status === 'downloading' || d.status === 'paused')) {
    return `<span class="badge badge-warn">${escHtml(d.status)} - see Downloads</span>`;
  }
  if (d && d.status === 'failed') {
    return '<span class="badge badge-warn">failed - retry it in Downloads</span>';
  }
  return `<button type="button" class="btn btn-secondary btn-sm btn-strata-model-download" data-model="${escHtml(m.name)}">
    <i class="fa-solid fa-download"></i> Download</button>`;
}

async function renderStrataModelsList(force = true) {
  const list = document.getElementById('strata-models-list');
  if (!list) return;
  const info = _engineInfo(await fetchEnginesForNode(_launchNode(), force), 'strata');
  if (!info) { list.innerHTML = '<p class="text-meta">Could not reach the target node.</p>'; return; }
  if (!info.available) {
    list.innerHTML = `<p class="text-meta">${escHtml(info.reason || 'Strata is unavailable on the target node.')}</p>`;
    return;
  }
  const src = info.catalogue || {};
  const source = src.source === 'image'
    ? '<p class="text-meta">Model list read from the installed Strata image.</p>'
    : `<p class="text-meta">Built-in model list${src.error ? ` (${escHtml(src.error)})` : ''}.</p>`;
  list.innerHTML = source + (info.models || []).map(m => `
    <div class="strata-model-row">
      <div class="strata-model-main">
        <div class="name">${escHtml(m.title || m.name)}</div>
        <div class="strata-model-badges">
          <span class="badge" title="Download size">${escHtml(m.size_display || '')}</span>
          ${m.ram_gb ? `<span class="badge" title="System RAM Strata needs">~${m.ram_gb} GB RAM</span>` : ''}
          ${m.vision ? '<span class="badge" title="Has an image encoder">images</span>' : ''}
          ${m.experimental ? '<span class="badge badge-warn">experimental</span>' : ''}
        </div>
        <span class="path">${escHtml(m.name)}</span>
      </div>
      <div class="strata-model-action">${_strataModelStatus(m)}</div>
    </div>`).join('');
  list.querySelectorAll('.btn-strata-model-download').forEach(btn => {
    btn.addEventListener('click', () => downloadStrataModel(btn.dataset.model, btn));
  });
}

async function downloadStrataModel(modelId, btn) {
  if (btn) btn.disabled = true;
  try {
    const res = await _nf(_launchNode(), '/api/engines/strata/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelId }),
    });
    const data = await readApiResponse(res);
    if (res.ok) {
      toast(`Downloading ${modelId} - progress in the Downloads panel`, 'success');
      if (typeof pollDownloads === 'function') pollDownloads();
    } else {
      toast(`Download failed: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error starting download: ' + e.message, 'error');
  } finally {
    renderStrataModelsList(true);
  }
}

function openStrataModelsModal() {
  document.getElementById('strata-models-modal')?.classList.add('open');
  renderStrataModelsList(true);
}

function closeStrataModelsModal() {
  document.getElementById('strata-models-modal')?.classList.remove('open');
}

// The button beside the Inference Engine dropdown: whenever the target node
// can run Strata, so a first model can be downloaded before any is local.
async function updateStrataModelsButton() {
  const btn = document.getElementById('btn-strata-models-open');
  if (!btn) return;
  const info = _engineInfo(await fetchEnginesForNode(_launchNode()), 'strata');
  btn.hidden = !(info && info.available);
}

document.getElementById('btn-strata-models-open')?.addEventListener('click', openStrataModelsModal);
updateStrataModelsButton();
document.getElementById('btn-close-strata-models')?.addEventListener('click', closeStrataModelsModal);
const _strataModelsModal = document.getElementById('strata-models-modal');
if (_strataModelsModal) _strataModelsModal.addEventListener('click', (e) => {
  if (e.target === _strataModelsModal) closeStrataModelsModal();
});

['f-memory-limit', 'f-idle-timeout'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('input', updateStrataWarnings);
});
