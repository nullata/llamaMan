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
let launchEngineModel = null;      // library entry of the selected virtual model
let _enginesCache = {};            // nodeId -> {at, data} from /api/engines
const _ENGINES_TTL_MS = 15000;

function currentLaunchEngine() { return launchEngine; }

function engineForModel(m) {
  return (m && m.engine) || 'llamacpp';
}

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
  launchEngineModel = (launchEngine !== 'llamacpp') ? (model || null) : null;
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
}

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

  updateStrataDownloadRow();
  updateStrataWarnings();
}

function updateStrataDownloadRow() {
  const m = launchEngineModel || {};
  const status = document.getElementById('strata-download-status');
  const btn = document.getElementById('btn-strata-download');
  if (!status || !btn) return;
  const d = m.download;
  btn.hidden = true;
  if (m.local_shards) {
    status.textContent = 'Model files are in llamaMan\'s models folder: the first start skips the download.';
  } else if (d && (d.status === 'downloading' || d.status === 'paused')) {
    status.textContent = `llamaMan is downloading the model files (${d.status}) - see the Downloads tab. Launch is blocked until it finishes.`;
  } else {
    status.textContent = (d && d.status === 'failed')
      ? 'The llamaMan download failed - retry it in the Downloads tab, or launch and let Strata download into its volume.'
      : 'Not downloaded by llamaMan. Launching lets Strata download into its own volume (progress shows on the instance card), or pre-download here to track it in the Downloads tab.';
    btn.hidden = !!(d && d.status === 'failed');
  }
}

function updateStrataWarnings() {
  const ul = document.getElementById('strata-warnings');
  if (!ul) return;
  if (launchEngine !== 'strata') { ul.innerHTML = ''; return; }
  const m = launchEngineModel || {};
  const warn = [];
  const memText = document.getElementById('f-memory-limit')?.value.trim();
  const memGb = memText ? _parseMemoryLimitGb(memText) : null;
  if (memText) {
    if (memGb != null && memGb < 64) {
      warn.push(`Memory limit ${memText} is below ~64 GB: Strata loads 32-62 GB and may be OOM-killed. Low-RAM mode is forced on under a limit.`);
    } else {
      warn.push('A memory limit forces Strata\'s Low-RAM mode on (its setup cannot see container limits).');
    }
  }
  const idle = parseInt(document.getElementById('f-idle-timeout')?.value, 10) || 0;
  if (idle > 0) {
    warn.push('Idle timeout works, but a cold start loads 32-62 GB and takes minutes; the request that wakes it waits that long.');
  }
  if (!m.local_shards && !(m.download && m.download.status === 'completed')) {
    warn.push(`The first start downloads ${m.size_display || '~70 GB'} and prepares the model before the server answers (up to STRATA_LOAD_TIMEOUT).`);
  }
  ul.innerHTML = warn.map(w => `<li><i class="fa-solid fa-triangle-exclamation"></i> ${escHtml(w)}</li>`).join('');
}

function applyStrataPresetToLaunchForm(p) {
  if (!p) return;
  const v = document.getElementById('f-strata-vision');
  if (v) v.value = ['no', 'yes', 'cpu'].includes(p.strata_vision) ? p.strata_vision : 'no';
  const kv = document.getElementById('f-strata-kv');
  if (kv) kv.value = ['', 'int8', 'q4_0', 'k8v4'].includes(p.strata_kv || '') ? (p.strata_kv || '') : '';
  const lr = document.getElementById('f-strata-low-ram');
  if (lr) lr.value = ['auto', 'on', 'off'].includes(p.strata_low_ram) ? p.strata_low_ram : 'auto';
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
  };
  if (typeof validateProxySamplingBody === 'function') validateProxySamplingBody(body);
  const memoryLimit = val('f-memory-limit').value.trim();
  if (memoryLimit) body.memory_limit = memoryLimit;
  const image = val('f-image')?.value;
  if (image) body.image = image;
  return body;
}

async function startStrataPredownload() {
  const m = launchEngineModel;
  if (!m) return;
  const btn = document.getElementById('btn-strata-download');
  if (btn) btn.disabled = true;
  try {
    const res = await _nf(_launchNode(), '/api/engines/strata/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: m.name }),
    });
    const data = await readApiResponse(res);
    if (res.ok) {
      toast(`Downloading ${m.name} (${m.size_display}) - see the Downloads tab`, 'success');
      m.download = { id: data.id, status: data.status };
      if (typeof pollDownloads === 'function') pollDownloads();
      if (typeof loadModels === 'function') loadModels();
    } else {
      toast(`Download failed: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error starting download: ' + e.message, 'error');
  } finally {
    if (btn) btn.disabled = false;
    updateStrataDownloadRow();
    updateStrataWarnings();
  }
}

const _strataDownloadBtn = document.getElementById('btn-strata-download');
if (_strataDownloadBtn) _strataDownloadBtn.addEventListener('click', startStrataPredownload);
['f-memory-limit', 'f-idle-timeout'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('input', updateStrataWarnings);
});
