// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Model metadata & GPU layer suggestion state
// -------------------------------------------------------------------------
let currentModelMeta = null;  // last fetched GGUF architecture metadata
let _gpuCache = null;         // cached GPU info array
let _gpuCacheTs = 0;          // timestamp of last GPU cache fill
let _suggestionTimer = null;  // debounce handle

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

async function fetchGpuInfoCached() {
  const now = Date.now();
  if (_gpuCache && now - _gpuCacheTs < 15000) return _gpuCache;
  try {
    const res = await apiFetch('/api/gpu-info');
    if (res && res.ok) {
      const data = await res.json();
      _gpuCache = data.gpus || [];
      _gpuCacheTs = now;
    }
  } catch (e) { /* ignore */ }
  return _gpuCache || [];
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
  const gpus = await fetchGpuInfoCached();
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

function renderModels() {
  const list = document.getElementById('model-list');
  if (!list) return;
  const query = document.getElementById('model-search').value.toLowerCase().trim();

  const filtered = query
    ? allModels.filter(m => m.name.toLowerCase().includes(query) || m.path.toLowerCase().includes(query) || (m.quant && m.quant.toLowerCase().includes(query)))
    : allModels;

  // Sort favorites first alphabetically, then the rest alphabetically
  filtered.sort((a, b) => {
    const favDiff = (isModelFavorited(a.path) ? 0 : 1) - (isModelFavorited(b.path) ? 0 : 1);
    if (favDiff !== 0) return favDiff;
    return a.name.localeCompare(b.name);
  });

  list.innerHTML = '';
  if (filtered.length === 0) {
    list.innerHTML = `<div id="model-empty">${allModels.length === 0 ? 'No models found in /models' : 'No matches'}</div>`;
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
  try {
    const res = await apiFetch(`/api/presets${encodePathForUrl(model.path)}`);
    if (res.ok) {
      const p = await res.json();
      if (p.n_gpu_layers != null) document.getElementById('f-gpu-layers').value = p.n_gpu_layers;
      if (p.ctx_size != null && ctxField) ctxField.value = p.ctx_size;
      document.getElementById('f-threads').value = p.threads || '';
      document.getElementById('f-memory-limit').value = p.memory_limit || '';
      document.getElementById('f-parallel').value = p.parallel || '';
      document.getElementById('f-extra').value = p.extra_args || '';
      document.getElementById('f-spec-enabled').checked = !!p.spec_enabled;
      document.getElementById('f-spec-draft-n-max').value = p.spec_draft_n_max ?? '';
      document.getElementById('f-gpu-devices').value = p.gpu_devices || '';
      document.getElementById('f-idle-timeout').value = p.idle_timeout_min || 0;
      document.getElementById('f-max-concurrent').value = p.max_concurrent || 0;
      document.getElementById('f-max-queue-depth').value = p.max_queue_depth || 200;
      document.getElementById('f-share-queue').checked = !!p.share_queue;
      document.getElementById('f-embedding-model').checked = !!p.embedding_model;
      document.getElementById('f-proxy-sampling-override-enabled').checked = !!p.proxy_sampling_override_enabled;
      document.getElementById('f-proxy-sampling-temperature').value = p.proxy_sampling_temperature ?? 0.8;
      document.getElementById('f-proxy-sampling-top-k').value = p.proxy_sampling_top_k ?? 40;
      document.getElementById('f-proxy-sampling-top-p').value = p.proxy_sampling_top_p ?? 0.95;
      document.getElementById('f-proxy-sampling-presence-penalty').value = p.proxy_sampling_presence_penalty ?? 0.0;
      document.getElementById('f-proxy-sampling-repeat-penalty').value = p.proxy_sampling_repeat_penalty ?? 0.0;
      document.getElementById('f-note').value = p.note || '';
      if (typeof updateProxySamplingOverrideState === 'function') updateProxySamplingOverrideState();
      if (typeof updateSpecState === 'function') updateSpecState();
      toast('Preset loaded', 'info');
    }
  } catch (e) { /* no preset, use defaults */ }
  // Detect layer count for model
  await updateGpuLayersTotal(model.path);
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
    const res = await apiFetch('/api/models/delete', {
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
    const res = await apiFetch(`/api/model-layers?path=${encodeURIComponent(modelPath)}`);
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
