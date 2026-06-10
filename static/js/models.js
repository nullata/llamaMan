// Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Model metadata & GPU layer suggestion state
// -------------------------------------------------------------------------
let currentModelMeta = null;  // last fetched GGUF architecture metadata
let _gpuCache = {};           // nodeId -> cached GPU info array
let _gpuCacheTs = {};         // nodeId -> timestamp of last cache fill
let _suggestionTimer = null;  // debounce handle
let _loadedPreset = null;     // last preset loaded into the form (for per-node hardware)
let _loadedPresetPath = null;

// Cluster-aware helpers (cluster.js loads after this file, so resolve at call time).
function _launchNode() {
  return (typeof getLaunchNode === 'function') ? getLaunchNode() : null;
}
function _nf(nodeId, path, opts) {
  return (typeof nodeFetch === 'function') ? nodeFetch(nodeId, path, opts) : apiFetch(path, opts);
}

// Approximate effective bits-per-weight for common GGUF quant types
const QUANT_BITS = {
  'F32': 32, 'F16': 16, 'BF16': 16,
  'Q8_0': 8,
  'Q6_K': 6.5,
  'Q5_K_M': 5.5, 'Q5_K_S': 5.5, 'Q5_K_L': 5.5, 'Q5_K': 5.5,
  'Q5_0': 5.0,   'Q5_1': 5.3,
  'Q4_K_M': 4.5, 'Q4_K_S': 4.25, 'Q4_K': 4.5,
  'Q4_0': 4.0,   'Q4_1': 4.3,
  'Q3_K_L': 3.6, 'Q3_K_M': 3.35, 'Q3_K_S': 3.0, 'Q3_K': 3.35,
  'Q2_K': 2.6,
  'IQ4_XS': 4.25, 'IQ4_NL': 4.5,
  'IQ3_XS': 3.3,  'IQ3_XXS': 3.06,
  'IQ2_XS': 2.31, 'IQ2_XXS': 2.06, 'IQ2_S': 2.5,
  'IQ1_S': 1.56,  'IQ1_M': 1.75,
};

async function fetchGpuInfoCached(nodeId) {
  const key = nodeId || 'local';
  const now = Date.now();
  if (_gpuCache[key] && now - (_gpuCacheTs[key] || 0) < 15000) return _gpuCache[key];
  try {
    const res = await _nf(nodeId, '/api/gpu-info');
    if (res && res.ok) {
      const data = await res.json();
      _gpuCache[key] = data.gpus || [];
      _gpuCacheTs[key] = now;
    }
  } catch (e) { /* ignore */ }
  return _gpuCache[key] || [];
}

/**
 * Estimate max transformer layers that fit in gpuFreeMb of VRAM.
 *
 * formula:
 *   layer_size    = params_per_layer * bits_per_weight / 8
 *   kv_per_layer  = 2 * n_kv_heads * head_dim * ctx_len * 2  (fp16)
 *   budget        = vram_free - overhead - non_layer_weights
 *   max_layers    = floor(budget / (layer_size + kv_per_layer))
 *
 * Dividing by (layer_size + kv_per_layer) avoids the circular dependency that
 * arises when subtracting total KV cache upfront: only GPU-resident layers'
 * KV cache actually lives in VRAM, so KV is a per-layer cost, not a fixed one.
 */
function calcMaxGpuLayers(meta, gpuFreeMb, ctxSize) {
  const { block_count, embedding_length, feed_forward_length, head_count, head_count_kv, quant } = meta;
  if (!block_count || !embedding_length || !feed_forward_length || !head_count) return null;

  const nKvHeads = head_count_kv || head_count;
  const headDim = Math.floor(embedding_length / head_count);
  const vocabSize = meta.vocab_size || 32000;
  const bitsPerWeight = QUANT_BITS[quant] ?? 4.5;

  // Per-layer weight params: Q + K + V + O attention projections + SwiGLU FFN (gate/up/down) + layer norms
  const attnParams = 2 * embedding_length * embedding_length
                   + 2 * nKvHeads * headDim * embedding_length;
  const ffnParams  = 3 * embedding_length * feed_forward_length;
  const normParams = 4 * embedding_length;  // negligible but included
  const layerSizeBytes = (attnParams + ffnParams + normParams) * bitsPerWeight / 8;

  // Non-layer weights: token embeddings + lm_head, assumed fp16
  const nonLayerBytes = 2 * vocabSize * embedding_length * 2;

  // KV cache per layer (fp16): only GPU-resident layers consume GPU VRAM,
  // so treat it as a per-layer cost alongside weight bytes (avoids circular dependency).
  const kvPerLayerBytes = 2 * nKvHeads * headDim * ctxSize * 2;

  // CUDA context + llama.cpp buffer overhead
  const OVERHEAD_BYTES = 512 * 1024 * 1024;

  // budget = vram_free - fixed costs; then divide by per-layer cost (weights + kv)
  const budget = gpuFreeMb * 1048576 - OVERHEAD_BYTES - nonLayerBytes;
  if (budget <= 0) return 0;
  return Math.max(0, Math.min(Math.floor(budget / (layerSizeBytes + kvPerLayerBytes)), block_count));
}

async function updateGpuLayersSuggestion() {
  const el = document.getElementById('gpu-layers-suggestion');
  if (!el) return;
  if (!currentModelMeta || !currentModelMeta.block_count) {
    el.textContent = '';
    el.classList.remove('text-success');
    return;
  }

  const ctxSize = parseInt(document.getElementById('f-ctx-size').value) || 4096;
  const gpuDevicesRaw = (document.getElementById('f-gpu-devices').value || '').trim();
  const gpus = await fetchGpuInfoCached(_launchNode());
  if (!gpus.length) {
    el.textContent = '';
    el.classList.remove('text-success');
    return;
  }

  // Determine which GPUs are active
  let activeGpus;
  if (gpuDevicesRaw) {
    const indices = gpuDevicesRaw.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    activeGpus = gpus.filter(g => indices.includes(g.index));
  } else {
    activeGpus = gpus;
  }
  if (!activeGpus.length) {
    el.textContent = '';
    el.classList.remove('text-success');
    return;
  }

  // For multi-GPU: llama.cpp distributes layers, so sum free VRAM
  const totalFreeMb = activeGpus.reduce((s, g) => s + g.memory_free_mb, 0);
  const maxLayers = calcMaxGpuLayers(currentModelMeta, totalFreeMb, ctxSize);
  if (maxLayers === null) {
    el.textContent = '';
    el.classList.remove('text-success');
    return;
  }

  const freeGb = (totalFreeMb / 1024).toFixed(1);
  const allFit = maxLayers >= currentModelMeta.block_count;
  const gpuLabel = activeGpus.length === 1
    ? `GPU ${activeGpus[0].index}`
    : `${activeGpus.length} GPUs`;

  el.textContent = `Suggested ≤${maxLayers} layers (${gpuLabel}, ${freeGb} GB free)${allFit ? ' - full offload fits' : ''}`;
  el.classList.toggle('text-success', allFit);
}

function scheduleSuggestionUpdate() {
  clearTimeout(_suggestionTimer);
  _suggestionTimer = setTimeout(updateGpuLayersSuggestion, 300);
}

// -------------------------------------------------------------------------
// Model metadata (favorites, notes) - stored in presets
// -------------------------------------------------------------------------
let allPresets = {};  // model_path -> preset object (loaded once, kept in sync)

async function loadAllPresets() {
  try {
    const res = await apiFetch('/api/presets');
    if (res && res.ok) allPresets = await res.json();
  } catch (e) { /* ignore */ }
}

function isModelFavorited(modelPath) {
  return !!(allPresets[modelPath] && allPresets[modelPath].favorite);
}

function getModelNote(modelPath) {
  return (allPresets[modelPath] && allPresets[modelPath].note) || '';
}

async function toggleFavorite(modelPath) {
  const newVal = !isModelFavorited(modelPath);
  if (!allPresets[modelPath]) allPresets[modelPath] = {};
  allPresets[modelPath].favorite = newVal;
  try {
    await apiFetch(`/api/presets${encodePathForUrl(modelPath)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ favorite: newVal }),
    });
  } catch (e) { /* ignore */ }
  return newVal;
}

async function saveModelNote(modelPath, note) {
  if (!allPresets[modelPath]) allPresets[modelPath] = {};
  allPresets[modelPath].note = note;
  try {
    await apiFetch(`/api/presets${encodePathForUrl(modelPath)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ note }),
    });
  } catch (e) { /* ignore */ }
}

// -------------------------------------------------------------------------
// Model Library
// -------------------------------------------------------------------------
async function loadModels() {
  const list = document.getElementById('model-list');
  try {
    await loadAllPresets();
    const res = await apiFetch('/api/models');
    allModels = await res.json();
    renderModels();
  } catch (e) {
    list.innerHTML = '<div id="model-empty">Error loading models</div>';
  }
}

// Build the present (launchable) and ghost (available on other nodes) model
// lists for the current launch Target node. Outside cluster mode this is just
// the local models with no ghosts.
function _modelSets() {
  const cs = window.clusterState;
  if (!(cs && cs.enabled)) return { present: allModels, ghost: [] };

  const target = _launchNode();
  const targetNode = (cs.nodes || []).find(n => n.node_id === target);
  const present = (target === cs.self_id)
    ? allModels
    : ((targetNode && targetNode.snapshot && targetNode.snapshot.models) || []);

  const presentKeys = new Set(present.map(m => m.name.toLowerCase()));
  const ghostMap = {};
  (cs.nodes || []).forEach(n => {
    if (n.node_id === target) return;
    ((n.snapshot && n.snapshot.models) || []).forEach(m => {
      const key = m.name.toLowerCase();
      if (presentKeys.has(key)) return;
      if (!ghostMap[key]) ghostMap[key] = { ...m, _onNodes: [] };
      ghostMap[key]._onNodes.push(n.node_name);
    });
  });
  return { present, ghost: Object.values(ghostMap) };
}

function renderModels() {
  const list = document.getElementById('model-list');
  if (!list) return;
  const query = document.getElementById('model-search').value.toLowerCase().trim();
  const matches = (m) => !query || m.name.toLowerCase().includes(query)
    || (m.path && m.path.toLowerCase().includes(query))
    || (m.quant && m.quant.toLowerCase().includes(query));

  const { present, ghost } = _modelSets();
  const filtered = present.filter(matches);
  const ghostFiltered = ghost.filter(matches);

  filtered.sort((a, b) => {
    const favDiff = (isModelFavorited(a.path) ? 0 : 1) - (isModelFavorited(b.path) ? 0 : 1);
    if (favDiff !== 0) return favDiff;
    return a.name.localeCompare(b.name);
  });
  ghostFiltered.sort((a, b) => a.name.localeCompare(b.name));

  list.innerHTML = '';
  if (filtered.length === 0 && ghostFiltered.length === 0) {
    list.innerHTML = `<div id="model-empty">${present.length === 0 && ghost.length === 0 ? 'No models found in /models' : 'No matches'}</div>`;
    return;
  }

  filtered.forEach(m => {
    const el = document.createElement('div');
    el.className = 'model-item' + (m.path === selectedModelPath ? ' selected' : '');
    const quantBadge = m.quant ? `<span class="badge badge-quant">${escHtml(m.quant)}</span>` : '';
    const fav = isModelFavorited(m.path);
    const starClass = fav ? 'btn-star active' : 'btn-star';
    const starIcon = fav ? 'fa-solid fa-star' : 'fa-regular fa-star';
    el.innerHTML = `
      <div class="model-item-row">
        <button class="${starClass}" title="Toggle favorite"><i class="${starIcon}"></i></button>
        <div class="model-item-content">
          <span class="name">${escHtml(m.name)}</span>
          <div class="badges">
            <span class="badge">${m.type.toUpperCase()}</span>
            ${quantBadge}
            <span class="badge badge-size">${escHtml(m.size_display)}</span>
          </div>
          <span class="path">${escHtml(m.path)}</span>
        </div>
      </div>
      <button class="btn-delete-model" title="Delete model from disk"><i class="fa-solid fa-trash"></i></button>
    `;
    el.querySelector('.btn-star').addEventListener('click', async (e) => {
      e.stopPropagation();
      await toggleFavorite(m.path);
      renderModels();
      updateLaunchFormStar();
    });
    el.querySelector('.btn-delete-model').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteModel(m);
    });
    el.addEventListener('click', () => selectModel(m, el));
    list.appendChild(el);
  });

  if (ghostFiltered.length === 0) return;

  const targetName = (typeof nodeNameById === 'function' && nodeNameById(_launchNode())) || 'this node';
  const divider = document.createElement('div');
  divider.className = 'model-ghost-divider';
  divider.textContent = `Available on other nodes`;
  list.appendChild(divider);

  ghostFiltered.forEach(m => {
    const el = document.createElement('div');
    el.className = 'model-item model-item-ghost';
    const quantBadge = m.quant ? `<span class="badge badge-quant">${escHtml(m.quant)}</span>` : '';
    el.innerHTML = `
      <div class="model-item-row">
        <div class="model-item-content">
          <span class="name">${escHtml(m.name)}</span>
          <div class="badges">
            <span class="badge">${(m.type || 'gguf').toUpperCase()}</span>
            ${quantBadge}
            <span class="badge badge-size">${escHtml(m.size_display || '')}</span>
          </div>
          <span class="path">on ${escHtml((m._onNodes || []).join(', '))}</span>
        </div>
      </div>
      <button class="btn-ghost-download" title="Download to ${escHtml(targetName)}"><i class="fa-solid fa-download"></i></button>
    `;
    el.querySelector('.btn-ghost-download').addEventListener('click', (e) => {
      e.stopPropagation();
      ghostDownload(m);
    });
    list.appendChild(el);
  });
}

// One-click: download a model that exists elsewhere onto the current Target
// node. Falls back to the download modal when the source has no known repo id.
async function ghostDownload(m) {
  const target = _launchNode();
  const targetName = (typeof nodeNameById === 'function' && nodeNameById(target)) || 'node';
  if (!m.repo_id) {
    if (typeof openDownloadModal === 'function') {
      openDownloadModal();
      const dn = document.getElementById('d-node');
      if (dn) dn.value = target;
      const repo = document.getElementById('d-repo-id');
      if (repo) repo.focus();
    }
    toast('No source repo recorded  fill in the download form', 'info');
    return;
  }
  const filename = (m.type === 'gguf' && m.path) ? m.path.split('/').pop() : '';
  try {
    const res = await _nf(target, '/api/downloads', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id: m.repo_id, filename }),
    });
    const data = await res.json();
    if (res.ok) {
      toast(`Downloading ${m.name} to ${targetName}`, 'success');
      if (typeof pollDownloads === 'function') pollDownloads();
    } else {
      toast(`Download failed: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error starting download: ' + e.message, 'error');
  }
}

async function selectModel(model, el) {
  document.querySelectorAll('.model-item').forEach(x => x.classList.remove('selected'));
  el.classList.add('selected');
  selectedModelPath = model.path;
  document.getElementById('f-model-path').value = model.path;
  document.getElementById('f-note').value = getModelNote(model.path);
  updateLaunchFormRepoInfo(model);
  updateLaunchFormStar();
  const ctxField = document.getElementById('f-ctx-size');
  if (typeof setActiveTab === 'function') setActiveTab('settings', 'launch');
  updatePortSuggestion();
  if (ctxField) ctxField.value = '';
  // Load preset if one exists
  _loadedPreset = null;
  _loadedPresetPath = null;
  try {
    const res = await apiFetch(`/api/presets${encodePathForUrl(model.path)}`);
    if (res.ok) {
      const p = await res.json();
      _loadedPreset = p;
      _loadedPresetPath = model.path;
      // Shared (cluster-wide) fields
      if (p.ctx_size != null && ctxField) ctxField.value = p.ctx_size;
      document.getElementById('f-extra').value = p.extra_args || '';
      document.getElementById('f-spec-enabled').checked = !!p.spec_enabled;
      document.getElementById('f-spec-draft-n-max').value = p.spec_draft_n_max ?? '';
      document.getElementById('f-idle-timeout').value = p.idle_timeout_min || 0;
      document.getElementById('f-max-concurrent').value = p.max_concurrent || 0;
      document.getElementById('f-max-queue-depth').value = p.max_queue_depth || 200;
      document.getElementById('f-share-queue').checked = !!p.share_queue;
      const _gIn = document.getElementById('f-share-queue-group');
      if (_gIn) _gIn.value = p.share_queue_group || '';
      const _fbIn = document.getElementById('f-share-queue-fallback');
      if (_fbIn) _fbIn.checked = !!p.share_queue_fallback;
      if (typeof updateShareQueueClusterRow === 'function') updateShareQueueClusterRow();
      document.getElementById('f-auto-restart').checked = !!p.auto_restart_on_crash;
      document.getElementById('f-embedding-model').checked = !!p.embedding_model;
      document.getElementById('f-proxy-sampling-override-enabled').checked = !!p.proxy_sampling_override_enabled;
      document.getElementById('f-proxy-sampling-temperature').value = p.proxy_sampling_temperature ?? 0.8;
      document.getElementById('f-proxy-sampling-top-k').value = p.proxy_sampling_top_k ?? 40;
      document.getElementById('f-proxy-sampling-top-p').value = p.proxy_sampling_top_p ?? 0.95;
      document.getElementById('f-proxy-sampling-presence-penalty').value = p.proxy_sampling_presence_penalty ?? 0.0;
      document.getElementById('f-proxy-sampling-repeat-penalty').value = p.proxy_sampling_repeat_penalty ?? 0.0;
      document.getElementById('f-note').value = p.note || '';
      // Per-node hardware (base, overlaid with the selected node's override)
      applyPresetHardwareForNode(p, _launchNode());
      if (typeof updateProxySamplingOverrideState === 'function') updateProxySamplingOverrideState();
      if (typeof updateSpecState === 'function') updateSpecState();
      toast('Preset loaded', 'info');
    }
  } catch (e) { /* no preset, use defaults */ }
  // Detect layer count for model
  await updateGpuLayersTotal(model.path);
}

// Populate the launch Docker-image dropdown from the target node's images,
// preserving the current pick and defaulting to that node's configured image.
async function populateLaunchImageSelect() {
  const sel = document.getElementById('f-image');
  if (!sel) return;
  const want = sel.value;  // preserve an explicit pick across refreshes
  try {
    const res = await _nf(_launchNode(), '/api/images');
    if (!res || !res.ok) return;
    const data = await res.json();
    const imgs = data.images || [];
    sel.innerHTML = '';
    imgs.forEach(img => {
      const opt = document.createElement('option');
      opt.value = img.name;
      const tags = [];
      if (img.name === data.current_image) tags.push('default');
      if (!img.present) tags.push('not pulled');
      opt.textContent = img.name + (tags.length ? `  (${tags.join(', ')})` : '');
      sel.appendChild(opt);
    });
    if (imgs.some(i => i.name === want)) sel.value = want;
    else if (data.current_image) sel.value = data.current_image;
  } catch (e) { /* ignore */ }
}

// Fill the hardware fields from a preset's base, overlaid with one node's
// override block (cluster mode). The shared fields are handled by the caller.
function applyPresetHardwareForNode(p, nodeId) {
  if (!p) return;
  const ov = (nodeId && p.node_overrides && p.node_overrides[nodeId]) || {};
  const val = (k) => (ov[k] !== undefined && ov[k] !== null) ? ov[k] : p[k];
  const layers = val('n_gpu_layers');
  if (layers != null) document.getElementById('f-gpu-layers').value = layers;
  document.getElementById('f-threads').value = val('threads') || '';
  document.getElementById('f-memory-limit').value = val('memory_limit') || '';
  document.getElementById('f-parallel').value = val('parallel') || '';
  document.getElementById('f-gpu-devices').value = val('gpu_devices') || '';
}

// Reset the launch form to a "no model selected" state. Used when switching
// the Target node: the new node's model library and presets are independent,
// so carrying over the previous node's fields (model path, ctx_size, share
// queue, sampling overrides, ...) is misleading - the user almost always has
// to pick a model on the new node anyway. We preserve only the node and image
// selects (the image select is repopulated by onLaunchNodeChanged immediately
// after). form.reset() clears values to their HTML defaults and unchecks
// checkboxes; we then clear the JS-side cached preset/model state so the next
// model selection starts clean.
function resetLaunchForm() {
  const form = document.getElementById('launch-form');
  if (!form) return;
  const node = document.getElementById('f-node')?.value;
  form.reset();
  if (node) {
    const sel = document.getElementById('f-node');
    if (sel) sel.value = node;
  }
  selectedModelPath = null;
  _loadedPreset = null;
  _loadedPresetPath = null;
  currentModelMeta = null;
  // form.reset() doesn't fire change events, so the share-queue cluster row
  // (which hides + clears its inputs on toggle-off) needs a manual nudge.
  if (typeof updateShareQueueClusterRow === 'function') updateShareQueueClusterRow();
  if (typeof updateLaunchFormRepoInfo === 'function') updateLaunchFormRepoInfo(null);
  if (typeof updateLaunchFormStar === 'function') updateLaunchFormStar();
  const total = document.getElementById('gpu-layers-total');
  if (total) total.textContent = '';
  const sugg = document.getElementById('gpu-layers-suggestion');
  if (sugg) { sugg.textContent = ''; sugg.classList.remove('text-success'); }
  if (typeof updateProxySamplingOverrideState === 'function') updateProxySamplingOverrideState();
  if (typeof updateSpecState === 'function') updateSpecState();
}

// Called when the launch Target node changes. Different node = different model
// library and (effectively) a fresh launch, so the form is reset; the user
// picks a model on the new node and its preset re-populates as normal.
async function onLaunchNodeChanged() {
  resetLaunchForm();
  renderModels();
  populateLaunchImageSelect();
  updatePortSuggestion();
}

function updateLaunchFormRepoInfo(model) {
  const el = document.getElementById('f-repo-info');
  if (!el) return;
  if (model && model.repo_id) {
    const repoId = escHtml(model.repo_id);
    el.innerHTML = `<i class="fa-solid fa-cube" style="margin-right:4px"></i><a href="https://huggingface.co/${encodeURI(model.repo_id)}" target="_blank" rel="noopener" title="${repoId}">${repoId}</a>`;
    el.hidden = false;
  } else {
    el.innerHTML = '';
    el.hidden = true;
  }
}

function updateLaunchFormStar() {
  const btn = document.getElementById('f-favorite');
  if (!btn) return;
  const modelPath = document.getElementById('f-model-path').value.trim();
  const fav = modelPath ? isModelFavorited(modelPath) : false;
  btn.classList.toggle('active', fav);
  btn.querySelector('i').className = fav ? 'fa-solid fa-star' : 'fa-regular fa-star';
}

async function deleteModel(model) {
  const ok = await showConfirm('Delete Model', `Delete "${model.name}" (${model.size_display}) from disk?\n\n${model.path}\n\nThis cannot be undone.`);
  if (!ok) return;
  try {
    const res = await _nf(_launchNode(), '/api/models/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: model.path }),
    });
    const data = await res.json();
    if (res.ok) {
      toast('Model deleted', 'info');
      if (selectedModelPath === model.path) selectedModelPath = null;
      await loadModels();
    } else {
      toast(`Cannot delete: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error deleting model: ' + e.message, 'error');
  }
}

async function updateGpuLayersTotal(modelPath) {
  const label = document.getElementById('gpu-layers-total');
  const suggEl = document.getElementById('gpu-layers-suggestion');
  label.textContent = '';
  currentModelMeta = null;
  if (suggEl) {
    suggEl.textContent = '';
    suggEl.classList.remove('text-success');
  }
  if (!modelPath || !modelPath.toLowerCase().endsWith('.gguf')) return;
  try {
    const res = await _nf(_launchNode(), `/api/model-layers?path=${encodeURIComponent(modelPath)}`);
    const data = await res.json();
    if (data.layers && data.layers > 0) {
      label.textContent = `/ ${data.layers}`;
    }
    currentModelMeta = data;
    await updateGpuLayersSuggestion();
  } catch (e) { /* ignore */ }
}

// Detect layers when model path is changed manually
const modelPathField = document.getElementById('f-model-path');
if (modelPathField) {
  modelPathField.addEventListener('change', function() {
    updateGpuLayersTotal(this.value.trim());
  });
}

// Re-compute suggestion when context size or GPU device selection changes
const ctxSizeField = document.getElementById('f-ctx-size');
if (ctxSizeField) ctxSizeField.addEventListener('input', scheduleSuggestionUpdate);

const gpuDevicesField = document.getElementById('f-gpu-devices');
if (gpuDevicesField) gpuDevicesField.addEventListener('input', scheduleSuggestionUpdate);

// Launch form star toggle
const launchStarBtn = document.getElementById('f-favorite');
if (launchStarBtn) launchStarBtn.addEventListener('click', async () => {
  const modelPath = document.getElementById('f-model-path').value.trim();
  if (!modelPath) { toast('Select a model first', 'error'); return; }
  await toggleFavorite(modelPath);
  updateLaunchFormStar();
  renderModels();
});

// Launch form note auto-save on blur
const noteField = document.getElementById('f-note');
if (noteField) noteField.addEventListener('blur', () => {
  const modelPath = document.getElementById('f-model-path').value.trim();
  if (modelPath) saveModelNote(modelPath, noteField.value.trim());
});
