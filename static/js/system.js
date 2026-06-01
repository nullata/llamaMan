// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// System info (CPU & RAM)
// -------------------------------------------------------------------------

async function loadSystemInfo() {
  // In cluster mode the System card is rendered per-node by cluster.js.
  if (window.__clusterActive) return;
  try {
    const card = document.getElementById('system-info-card');
    const container = document.getElementById('system-info-bars');
    if (!card || !container) return;
    const res = await apiFetch('/api/system-info');
    const d = await res.json();
    if (d.error) return;

    const coresLabel = document.getElementById('system-cores');
    coresLabel.textContent = `${d.cpu_cores} cores`;

    const cpuPct = Math.round(d.cpu_percent);
    const ramPct = Math.round(d.ram_percent);
    const cpuColor = cpuPct > 90 ? 'var(--red)' : cpuPct > 70 ? 'var(--yellow)' : 'var(--green)';
    const ramColor = ramPct > 90 ? 'var(--red)' : ramPct > 70 ? 'var(--yellow)' : 'var(--green)';

    const ramUsedGB = (d.ram_used_mb / 1024).toFixed(1);
    const ramTotalGB = (d.ram_total_mb / 1024).toFixed(1);

    container.innerHTML = `
      <div class="gpu-bar-row">
        <span class="gpu-bar-label">CPU</span>
        <div class="gpu-bar-track">
          <div class="gpu-bar-fill" style="width:${cpuPct}%;background:${cpuColor};"></div>
        </div>
        <span class="gpu-bar-text">${cpuPct}%</span>
      </div>
      <div class="gpu-bar-row">
        <span class="gpu-bar-label">RAM</span>
        <div class="gpu-bar-track">
          <div class="gpu-bar-fill" style="width:${ramPct}%;background:${ramColor};"></div>
        </div>
        <span class="gpu-bar-text">${ramUsedGB} / ${ramTotalGB} GB (${ramPct}%)</span>
      </div>
    `;
  } catch (e) {
    // hide on error
  }
}

// -------------------------------------------------------------------------
// GPU VRAM indicator
// -------------------------------------------------------------------------
async function loadGpuInfo() {
  // In cluster mode the GPU card is rendered per-node by cluster.js.
  if (window.__clusterActive) return;
  try {
    const card = document.getElementById('gpu-vram-card');
    const container = document.getElementById('gpu-vram-bars');
    if (!card || !container) return;
    const res = await apiFetch('/api/gpu-info');
    const data = await res.json();

    if (!data.gpus || data.gpus.length === 0) {
      card.style.display = 'none';
      showGpuWarning();
      return;
    }

    hideGpuWarning();
    container.innerHTML = '';

    data.gpus.forEach(gpu => {
      const vramPct = Math.round((gpu.memory_used_mb / gpu.memory_total_mb) * 100);
      const vramColor = vramPct > 90 ? 'var(--red)' : vramPct > 70 ? 'var(--yellow)' : 'var(--green)';
      const corePct = gpu.utilization_pct ?? 0;
      const coreColor = corePct > 90 ? 'var(--red)' : corePct > 70 ? 'var(--yellow)' : 'var(--green)';
      const tempVal = Number.isFinite(gpu.temperature_c) ? gpu.temperature_c : null;
      const tempColor = tempVal == null ? 'var(--muted)'
        : tempVal >= 85 ? 'var(--red)'
        : tempVal >= 75 ? 'var(--yellow)'
        : 'var(--muted)';
      const tempHtml = tempVal == null
        ? '<span class="gpu-bar-temp" style="color:var(--muted);">-</span>'
        : `<span class="gpu-bar-temp" style="color:${tempColor};">${tempVal}&deg;C</span>`;
      const row = document.createElement('div');
      row.className = 'gpu-bar-row';
      row.innerHTML = `
        <div class="gpu-bar-label-col" title="${escHtml(gpu.name)}">
          <span class="gpu-bar-label">GPU ${gpu.index}</span>
          ${tempHtml}
        </div>
        <div style="flex:1;display:flex;flex-direction:column;gap:3px;">
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:0.75em;width:3em;color:var(--muted);">core</span>
            <div class="gpu-bar-track" style="flex:1;">
              <div class="gpu-bar-fill" style="width:${corePct}%;background:${coreColor};"></div>
            </div>
            <span class="gpu-bar-text" style="width:3.5em;">${corePct}%</span>
          </div>
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:0.75em;width:3em;color:var(--muted);">VRAM</span>
            <div class="gpu-bar-track" style="flex:1;">
              <div class="gpu-bar-fill" style="width:${vramPct}%;background:${vramColor};"></div>
            </div>
            <span class="gpu-bar-text" style="width:3.5em;">${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB</span>
          </div>
        </div>
      `;
      container.appendChild(row);
    });
  } catch (e) {
    const card = document.getElementById('gpu-vram-card');
    if (card) card.style.display = 'none';
    showGpuWarning();
  }
}
function showGpuWarning() {
  const warning = document.getElementById('gpu-warning');
  if (warning) warning.hidden = false;
}

function hideGpuWarning() {
  const warning = document.getElementById('gpu-warning');
  if (warning) warning.hidden = true;
}

// -------------------------------------------------------------------------
// Cleanup Settings
// -------------------------------------------------------------------------
async function loadSettings() {
  try {
    if (!document.getElementById('s-dl-cleanup-enabled')) return;
    const res = await apiFetch('/api/settings');
    if (!res) return;
    const s = await res.json();
    const c = s.cleanup || {};
    document.getElementById('s-dl-cleanup-enabled').checked = !!c.downloads_enabled;
    document.getElementById('s-dl-cleanup-age').value = c.downloads_max_age_hours ?? 24;
    document.getElementById('s-inst-cleanup-enabled').checked = !!c.instances_enabled;
    document.getElementById('s-inst-cleanup-age').value = c.instances_max_age_hours ?? 24;
    const orphanContainers = document.getElementById('s-orphan-containers-enabled');
    if (orphanContainers) orphanContainers.checked = !!c.orphan_containers_enabled;
    renderCleanupLastRan('s-dl-cleanup-last-ran', c.downloads_last_run_at);
    renderCleanupLastRan('s-inst-cleanup-last-ran', c.instances_last_run_at);
    renderCleanupLastRan('s-orphan-containers-last-ran', c.orphan_containers_last_run_at);

    const authToggle = document.getElementById('s-require-auth');
    if (authToggle) {
      authToggle.checked = s.require_auth !== false; // default ON
      updateAuthHint();
    }

    const speedLimit = document.getElementById('s-global-speed-limit');
    if (speedLimit) speedLimit.value = s.global_speed_limit_mbps ?? 0;
    const autoRetryFailedDownloads = document.getElementById('s-auto-retry-failed-downloads');
    if (autoRetryFailedDownloads) autoRetryFailedDownloads.checked = !!s.auto_retry_failed_downloads;
    const retryCountPerFailedDownload = document.getElementById('s-retry-count-per-failed-download');
    if (retryCountPerFailedDownload) {
      retryCountPerFailedDownload.value = s.retry_count_per_failed_download ?? 3;
      updateDownloadRetryCountState();
    }

    const adminUiEvictionToggle = document.getElementById('s-admin-ui-enforce-max-models');
    if (adminUiEvictionToggle) {
      adminUiEvictionToggle.checked = !!s.admin_ui_enforce_max_models;
      updateAdminUiEvictionHint();
    }
    const ollamaOverrideToggle = document.getElementById('s-allow-ollama-override-admin');
    if (ollamaOverrideToggle) ollamaOverrideToggle.checked = !!s.allow_ollama_api_override_admin;

    const recordingModeSelect = document.getElementById('s-recording-mode');
    if (recordingModeSelect) {
      const mode = s.recording_mode === 'per_request' || s.recording_mode === 'per_conversation'
        ? s.recording_mode : 'off';
      recordingModeSelect.value = mode;
    }
    const recordingRetentionInput = document.getElementById('s-recording-retention-days');
    if (recordingRetentionInput) {
      const days = Number.isFinite(Number(s.recording_retention_days)) ? Number(s.recording_retention_days) : 30;
      recordingRetentionInput.value = String(Math.max(0, days));
    }

    const staleEnabled = document.getElementById('s-stale-records-enabled');
    if (staleEnabled) staleEnabled.checked = !!c.stale_records_enabled;
    const staleInterval = document.getElementById('s-stale-records-interval');
    if (staleInterval) staleInterval.value = c.stale_records_interval_min ?? 5;
    renderCleanupLastRan('s-stale-records-last-ran', c.stale_records_last_run_at);

    await loadHuggingFaceTokens();
  } catch (e) {}
}

async function refreshCleanupLastRan() {
  try {
    const dlLabel = document.getElementById('s-dl-cleanup-last-ran');
    const instLabel = document.getElementById('s-inst-cleanup-last-ran');
    if (!dlLabel && !instLabel) return;
    const res = await apiFetch('/api/settings');
    if (!res) return;
    const s = await res.json();
    const c = s.cleanup || {};
    renderCleanupLastRan('s-dl-cleanup-last-ran', c.downloads_last_run_at);
    renderCleanupLastRan('s-inst-cleanup-last-ran', c.instances_last_run_at);
    renderCleanupLastRan('s-orphan-containers-last-ran', c.orphan_containers_last_run_at);
  } catch (e) {}
}

async function saveSettings() {
  if (!document.getElementById('s-dl-cleanup-enabled')) return;
  const payload = {
    cleanup: {
      downloads_enabled: document.getElementById('s-dl-cleanup-enabled').checked,
      downloads_max_age_hours: parseInt(document.getElementById('s-dl-cleanup-age').value) || 24,
      instances_enabled: document.getElementById('s-inst-cleanup-enabled').checked,
      instances_max_age_hours: parseInt(document.getElementById('s-inst-cleanup-age').value) || 24,
      orphan_containers_enabled: document.getElementById('s-orphan-containers-enabled').checked,
      stale_records_enabled: document.getElementById('s-stale-records-enabled').checked,
      stale_records_interval_min: parseInt(document.getElementById('s-stale-records-interval').value) || 5,
    }
  };
  try {
    const res = await apiFetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res && res.ok) {
      const status = document.getElementById('settings-status');
      status.textContent = 'Saved.';
      setTimeout(() => { status.textContent = ''; }, 2000);
    }
  } catch (e) {}
}

function renderCleanupLastRan(elementId, ts) {
  const el = document.getElementById(elementId);
  if (!el) return;
  if (!ts) {
    el.textContent = 'Never';
    return;
  }
  const date = new Date(ts * 1000);
  if (Number.isNaN(date.getTime())) {
    el.textContent = 'Never';
    return;
  }
  el.textContent = date.toLocaleString();
}

function updateAuthHint() {
  const hint = document.getElementById('auth-hint');
  const toggle = document.getElementById('s-require-auth');
  if (!hint || !toggle) return;
  hint.textContent = toggle.checked
    ? 'All API requests (including model loading) require a valid bearer token.'
    : 'Model loading endpoints are open. Only management endpoints require authentication.';
}

function updateAdminUiEvictionHint() {
  const hint = document.getElementById('admin-ui-eviction-hint');
  const toggle = document.getElementById('s-admin-ui-enforce-max-models');
  if (!hint || !toggle) return;
  hint.textContent = toggle.checked
    ? 'Admin UI launches will evict the least-recently-used non-embedding model when the cap is full.'
    : 'Admin UI launches can go beyond the cap after a confirmation prompt instead of evicting an existing model.';
}

async function saveRequireAuth() {
  const toggle = document.getElementById('s-require-auth');
  if (!toggle) return;
  try {
    const res = await apiFetch('/api/settings');
    if (!res) return;
    const current = await res.json();
    current.require_auth = toggle.checked;
    const save = await apiFetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(current),
    });
    if (save && save.ok) {
      updateAuthHint();
      toast(toggle.checked ? 'Authentication required for all endpoints' : 'Model endpoints are now open', 'info');
    }
  } catch (e) {
    toast('Error saving auth setting: ' + e.message, 'error');
  }
}

async function saveAppSettings() {
  const adminToggle = document.getElementById('s-admin-ui-enforce-max-models');
  const ollamaToggle = document.getElementById('s-allow-ollama-override-admin');
  const recordingModeSelect = document.getElementById('s-recording-mode');
  const recordingRetentionInput = document.getElementById('s-recording-retention-days');
  if (!adminToggle && !ollamaToggle && !recordingModeSelect && !recordingRetentionInput) return;
  try {
    const payload = {};
    if (adminToggle) payload.admin_ui_enforce_max_models = adminToggle.checked;
    if (ollamaToggle) payload.allow_ollama_api_override_admin = ollamaToggle.checked;
    if (recordingModeSelect) payload.recording_mode = recordingModeSelect.value;
    if (recordingRetentionInput) {
      const n = parseInt(recordingRetentionInput.value, 10);
      payload.recording_retention_days = Number.isFinite(n) && n >= 0 ? n : 30;
    }
    const res = await apiFetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res && res.ok) {
      updateAdminUiEvictionHint();
      toast('App settings saved', 'info');
    }
  } catch (e) {
    toast('Error saving app settings: ' + e.message, 'error');
  }
}

function buildModelsExportFilename() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, '0');
  const stamp = [
    now.getFullYear(),
    pad(now.getMonth() + 1),
    pad(now.getDate()),
  ].join('') + '-' + [
    pad(now.getHours()),
    pad(now.getMinutes()),
    pad(now.getSeconds()),
  ].join('');
  return `llamaman-models-${stamp}.json`;
}

async function downloadStoredModelsJson() {
  const button = document.getElementById('btn-download-models-json');
  if (!button) return;

  const originalHtml = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Preparing...';

  try {
    const [modelsRes, presetsRes] = await Promise.all([
      apiFetch('/api/models'),
      apiFetch('/api/presets'),
    ]);
    const models = await readApiResponse(modelsRes);
    if (!modelsRes || !modelsRes.ok) {
      throw new Error(models.error || 'Unable to load models');
    }
    const presets = presetsRes && presetsRes.ok ? await presetsRes.json() : {};

    const exported = models.map(({ path, size_bytes, size_display, ...rest }) => {
      const preset = presets[path];
      return preset ? { ...rest, preset } : rest;
    });

    const blob = new Blob([JSON.stringify(exported, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = buildModelsExportFilename();
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    toast(`Downloaded ${exported.length} stored models`, 'info');
  } catch (e) {
    toast('Error downloading stored models JSON: ' + e.message, 'error');
  } finally {
    button.disabled = false;
    button.innerHTML = originalHtml;
  }
}

const requireAuthToggle = document.getElementById('s-require-auth');
if (requireAuthToggle) {
  requireAuthToggle.addEventListener('change', saveRequireAuth);
}

const adminUiEvictionToggle = document.getElementById('s-admin-ui-enforce-max-models');
if (adminUiEvictionToggle) {
  adminUiEvictionToggle.addEventListener('change', saveAppSettings);
}

const ollamaOverrideToggle = document.getElementById('s-allow-ollama-override-admin');
if (ollamaOverrideToggle) {
  ollamaOverrideToggle.addEventListener('change', saveAppSettings);
}

const recordingModeSelect = document.getElementById('s-recording-mode');
if (recordingModeSelect) {
  recordingModeSelect.addEventListener('change', saveAppSettings);
}

const recordingRetentionInput = document.getElementById('s-recording-retention-days');
if (recordingRetentionInput) {
  recordingRetentionInput.addEventListener('change', saveAppSettings);
}

const downloadModelsJsonBtn = document.getElementById('btn-download-models-json');
if (downloadModelsJsonBtn) {
  downloadModelsJsonBtn.addEventListener('click', downloadStoredModelsJson);
}

const saveSettingsBtn = document.getElementById('btn-save-settings');
if (saveSettingsBtn) saveSettingsBtn.addEventListener('click', saveSettings);

// -------------------------------------------------------------------------
// Hugging Face Tokens
// -------------------------------------------------------------------------
function populateDownloadTokenOptions() {
  const select = document.getElementById('d-token-id');
  const hint = document.getElementById('d-token-hint');
  if (!select) return;

  const selected = select.value;
  select.innerHTML = '<option value="">No token</option>';
  huggingFaceTokens.forEach(token => {
    const option = document.createElement('option');
    option.value = token.id;
    option.textContent = `${token.name} (${token.preview})`;
    select.appendChild(option);
  });

  if (huggingFaceTokens.some(token => token.id === selected)) {
    select.value = selected;
  }

  if (hint) {
    hint.textContent = huggingFaceTokens.length > 0
      ? 'Pick a saved token for gated or private repos, or leave this on "No token".'
      : 'Use the Hugging Face tab in Settings to save tokens for gated or private repos.';
  }
}

async function loadHuggingFaceTokens() {
  const list = document.getElementById('hf-tokens-list');
  try {
    const res = await apiFetch('/api/settings/huggingface-tokens');
    if (!res) return;
    huggingFaceTokens = await res.json();
  } catch (e) {
    huggingFaceTokens = [];
  }

  populateDownloadTokenOptions();
  if (!list) return;

  if (huggingFaceTokens.length === 0) {
    list.innerHTML = '<div class="list-empty-state">No Hugging Face tokens saved.</div>';
    return;
  }

  list.innerHTML = '';
  huggingFaceTokens.forEach(token => {
    const created = token.created_at
      ? new Date(typeof token.created_at === 'number'
          ? token.created_at * 1000
          : token.created_at).toLocaleDateString()
      : '';
    const item = document.createElement('div');
    item.className = 'dl-item';
    item.innerHTML = `
      <div class="dl-item-top">
        <span class="dl-item-name"><strong>${escHtml(token.name)}</strong></span>
        <code class="list-meta-code">${escHtml(token.preview)}</code>
        <span class="list-meta-date">${escHtml(created)}</span>
        <button class="btn-xs danger btn-hf-token-delete" data-id="${token.id}"><i class="fa-solid fa-trash"></i> Delete</button>
      </div>
    `;
    list.appendChild(item);
  });

  list.querySelectorAll('.btn-hf-token-delete').forEach(btn => {
    btn.addEventListener('click', () => deleteHuggingFaceToken(btn.dataset.id));
  });
}

async function createHuggingFaceToken() {
  const nameInput = document.getElementById('hf-token-name');
  const valueInput = document.getElementById('hf-token-value');
  if (!nameInput || !valueInput) return;

  const name = nameInput.value.trim() || 'Untitled';
  const token = valueInput.value.trim();
  if (!token) {
    toast('Token value is required', 'error');
    return;
  }

  try {
    const res = await apiFetch('/api/settings/huggingface-tokens', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, token }),
    });
    const data = await res.json();
    if (res.ok) {
      toast('Hugging Face token saved', 'success');
      nameInput.value = '';
      valueInput.value = '';
      await loadHuggingFaceTokens();
    } else {
      toast(`Failed: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error saving token: ' + e.message, 'error');
  }
}

async function deleteHuggingFaceToken(id) {
  const ok = await showConfirm('Delete Hugging Face Token', 'Delete this saved Hugging Face token?');
  if (!ok) return;
  try {
    const res = await apiFetch(`/api/settings/huggingface-tokens/${id}`, { method: 'DELETE' });
    if (res.ok) {
      toast('Hugging Face token deleted', 'info');
      await loadHuggingFaceTokens();
    } else {
      const data = await res.json();
      toast(`Failed: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error deleting token: ' + e.message, 'error');
  }
}

const btnSaveHfToken = document.getElementById('btn-save-hf-token');
if (btnSaveHfToken) btnSaveHfToken.addEventListener('click', createHuggingFaceToken);

// -------------------------------------------------------------------------
// API Keys
// -------------------------------------------------------------------------
async function loadApiKeys() {
  const list = document.getElementById('api-keys-list');
  if (!list) return;
  try {
    const res = await apiFetch('/api/api-keys');
    const keys = await res.json();
    if (keys.length === 0) {
      list.innerHTML = '<div class="list-empty-state">No API keys yet. API is open to all requests.</div>';
      return;
    }
    list.innerHTML = '';
    keys.forEach(k => {
      const date = k.created_at
        ? new Date(typeof k.created_at === 'number'
            ? k.created_at * 1000
            : k.created_at).toLocaleDateString()
        : '';
      const item = document.createElement('div');
      item.className = 'dl-item';
      item.innerHTML = `
        <div class="dl-item-top">
          <span class="dl-item-name"><strong>${escHtml(k.name)}</strong></span>
          <code class="list-meta-code">${escHtml(k.prefix)}</code>
          <span class="list-meta-date">${date}</span>
          <button class="btn-xs danger btn-ak-delete" data-id="${k.id}"><i class="fa-solid fa-trash"></i> Revoke</button>
        </div>
      `;
      list.appendChild(item);
    });
    list.querySelectorAll('.btn-ak-delete').forEach(btn => {
      btn.addEventListener('click', () => revokeApiKey(btn.dataset.id));
    });
  } catch (e) { /* ignore */ }
}

async function createApiKey() {
  const nameInput = document.getElementById('ak-name');
  const name = nameInput.value.trim() || 'Untitled';
  try {
    const res = await apiFetch('/api/api-keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const data = await res.json();
    if (res.ok) {
      toast('API key created', 'success');
      nameInput.value = '';
      // Show the key in a modal
      document.getElementById('api-key-value').textContent = data.key;
      document.getElementById('api-key-modal').classList.add('open');
      await loadApiKeys();
    } else {
      toast(`Failed: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error creating key: ' + e.message, 'error');
  }
}

async function revokeApiKey(id) {
  const ok = await showConfirm('Revoke API Key', 'Revoke this API key? Any tools using it will lose access.');
  if (!ok) return;
  try {
    const res = await apiFetch(`/api/api-keys/${id}`, { method: 'DELETE' });
    if (res.ok) {
      toast('API key revoked', 'info');
      await loadApiKeys();
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

const btnCreateKey = document.getElementById('btn-create-api-key');
if (btnCreateKey) btnCreateKey.addEventListener('click', createApiKey);

const btnCopyKey = document.getElementById('btn-copy-api-key');
if (btnCopyKey) btnCopyKey.addEventListener('click', () => {
  const key = document.getElementById('api-key-value').textContent;
  const icon = btnCopyKey.querySelector('i');

  const onCopied = () => {
    if (icon) {
      icon.className = 'fa-solid fa-check';
      icon.classList.add('icon-copy-success');
      setTimeout(() => { icon.className = 'fa-solid fa-copy'; }, 2000);
    }
  };

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(key).then(onCopied);
  } else {
    const ta = document.createElement('textarea');
    ta.className = 'clipboard-copy-buffer';
    ta.value = key;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    onCopied();
  }
});

const apiKeyModal = document.getElementById('api-key-modal');
const btnCloseApiKeyModal = document.getElementById('btn-close-api-key-modal');
if (btnCloseApiKeyModal) btnCloseApiKeyModal.addEventListener('click', () => apiKeyModal.classList.remove('open'));
if (apiKeyModal) apiKeyModal.addEventListener('click', (e) => {
  if (e.target === apiKeyModal) apiKeyModal.classList.remove('open');
});

// -------------------------------------------------------------------------
// Download Settings (global speed limit)
// -------------------------------------------------------------------------
function updateDownloadRetryCountState() {
  const toggle = document.getElementById('s-auto-retry-failed-downloads');
  const input = document.getElementById('s-retry-count-per-failed-download');
  if (!toggle || !input) return;
  input.disabled = !toggle.checked;
}

async function saveDownloadSettings() {
  const limitMbps = parseFloat(document.getElementById('s-global-speed-limit').value) || 0;
  const autoRetryFailedDownloads = !!document.getElementById('s-auto-retry-failed-downloads')?.checked;
  const retryCountPerFailedDownload = Math.max(
    1,
    parseInt(document.getElementById('s-retry-count-per-failed-download').value, 10) || 3,
  );
  try {
    const res = await apiFetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        global_speed_limit_mbps: limitMbps,
        auto_retry_failed_downloads: autoRetryFailedDownloads,
        retry_count_per_failed_download: retryCountPerFailedDownload,
      }),
    });
    if (res && res.ok) {
      const status = document.getElementById('download-settings-status');
      status.textContent = limitMbps > 0 ? `Saved - ${limitMbps} Mbps limit active` : 'Saved - unlimited';
      setTimeout(() => { status.textContent = ''; }, 3000);
    }
  } catch (e) {
    toast('Error saving download settings: ' + e.message, 'error');
  }
}

const btnSaveDownloadSettings = document.getElementById('btn-save-download-settings');
if (btnSaveDownloadSettings) btnSaveDownloadSettings.addEventListener('click', saveDownloadSettings);
const autoRetryFailedDownloadsToggle = document.getElementById('s-auto-retry-failed-downloads');
if (autoRetryFailedDownloadsToggle) {
  autoRetryFailedDownloadsToggle.addEventListener('change', updateDownloadRetryCountState);
  updateDownloadRetryCountState();
}

// -------------------------------------------------------------------------
// Docker Image Management
// -------------------------------------------------------------------------

let _pullStatusInterval = null;

async function loadImages() {
  const list = document.getElementById('images-list');
  if (!list) return;
  try {
    const res = await nodeFetch(getImagesNode(), '/api/images');
    if (!res) return;
    const data = await res.json();

    const autoToggle = document.getElementById('s-image-auto-update');
    if (autoToggle) autoToggle.checked = !!data.auto_update_enabled;
    const intervalInput = document.getElementById('s-image-update-interval');
    if (intervalInput) intervalInput.value = data.auto_update_interval_hours ?? 24;

    if (!data.images || data.images.length === 0) {
      list.innerHTML = '<div class="list-empty-state">No images tracked yet.</div>';
      return;
    }

    list.innerHTML = '';
    data.images.forEach(img => {
      const isActive = img.name === data.current_image;
      const digestShort = img.digest ? img.digest.replace('sha256:', '').slice(0, 12) : '-';
      const sizeMb = img.size_mb ? `${img.size_mb} MB` : '-';
      const pulledAt = img.last_pulled_at
        ? new Date(img.last_pulled_at * 1000).toLocaleString()
        : 'Never';
      const presentBadge = img.present
        ? '<span class="badge badge-ok">local</span>'
        : '<span class="badge badge-warn">not pulled</span>';
      const defaultBadge = isActive ? '<span class="badge badge-info">default</span>' : '';

      const item = document.createElement('div');
      item.className = 'dl-item';
      item.innerHTML = `
        <div class="dl-item-top">
          <span class="dl-item-name"><strong>${escHtml(img.name)}</strong> ${defaultBadge} ${presentBadge}</span>
          <code class="list-meta-code" title="${escHtml(img.digest || '')}">${escHtml(digestShort)}</code>
          <span class="list-meta-date">${escHtml(sizeMb)}</span>
          <span class="list-meta-date">pulled: ${escHtml(pulledAt)}</span>
          <button class="btn-xs btn-image-pull" data-image="${escHtml(img.name)}">
            <i class="fa-solid fa-arrow-down-to-line"></i> Pull
          </button>
          <button class="btn-xs danger btn-image-delete" data-image="${escHtml(img.name)}" title="Delete (blocked while a running instance uses it)">
            <i class="fa-solid fa-trash"></i>
          </button>
        </div>
      `;
      list.appendChild(item);
    });

    list.querySelectorAll('.btn-image-pull').forEach(btn => {
      btn.addEventListener('click', () => triggerImagePull(btn.dataset.image));
    });
    list.querySelectorAll('.btn-image-delete').forEach(btn => {
      btn.addEventListener('click', () => deleteImage(btn.dataset.image));
    });
    // Keep the launch-form image dropdown in sync with available images.
    if (typeof populateLaunchImageSelect === 'function') populateLaunchImageSelect();
  } catch (e) {
    const list = document.getElementById('images-list');
    if (list) list.innerHTML = '<div class="list-empty-state">Error loading images.</div>';
  }
}

async function triggerImagePull(imageName) {
  const statusEl = document.getElementById('image-pull-status');
  try {
    const res = await nodeFetch(getImagesNode(), '/api/images/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageName }),
    });
    const data = await res.json();
    if (!res.ok) {
      toast(`Pull failed: ${data.error}`, 'error');
      return;
    }
    toast(`Pulling ${imageName}…`, 'info');
    if (statusEl) {
      statusEl.hidden = false;
      statusEl.textContent = 'Pulling…';
    }
    startPullStatusPolling();
  } catch (e) {
    toast('Error starting pull: ' + e.message, 'error');
  }
}

function startPullStatusPolling() {
  if (_pullStatusInterval) return;
  _pullStatusInterval = setInterval(pollPullStatus, 2000);
}

async function pollPullStatus() {
  const statusEl = document.getElementById('image-pull-status');
  try {
    const res = await nodeFetch(getImagesNode(), '/api/images/pull-status');
    if (!res) return;
    const state = await res.json();

    if (state.status === 'pulling') {
      if (statusEl) {
        statusEl.hidden = false;
        statusEl.textContent = `Pulling ${state.image}: ${state.message}`;
      }
      return;
    }

    // Done or error - stop polling
    clearInterval(_pullStatusInterval);
    _pullStatusInterval = null;

    if (state.status === 'done') {
      if (statusEl) statusEl.hidden = true;
      toast(`Image updated: ${state.image}`, 'success');
      await loadImages();
    } else if (state.status === 'error') {
      if (statusEl) {
        statusEl.hidden = false;
        statusEl.textContent = `Pull error: ${state.message}`;
      }
      toast(`Pull error: ${state.message}`, 'error');
    } else {
      if (statusEl) statusEl.hidden = true;
    }
  } catch (e) { /* ignore */ }
}

async function saveImageSettings() {
  const autoToggle = document.getElementById('s-image-auto-update');
  const intervalInput = document.getElementById('s-image-update-interval');
  if (!autoToggle || !intervalInput) return;

  try {
    const res = await nodeFetch(getImagesNode(), '/api/images/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        auto_update_enabled: autoToggle.checked,
        auto_update_interval_hours: parseInt(intervalInput.value) || 24,
      }),
    });
    if (res && res.ok) {
      const status = document.getElementById('image-settings-status');
      if (status) {
        status.textContent = 'Saved.';
        setTimeout(() => { status.textContent = ''; }, 2000);
      }
    }
  } catch (e) {
    toast('Error saving image settings: ' + e.message, 'error');
  }
}

async function deleteImage(imageName) {
  if (!confirm(`Remove image "${imageName}" from Docker?\n\nThis deletes the local image. You can re-pull it at any time.`)) return;
  try {
    const res = await nodeFetch(getImagesNode(), '/api/images', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageName }),
    });
    const data = await res.json();
    if (!res.ok) {
      toast(`Remove failed: ${data.error}`, 'error');
      return;
    }
    toast(`Removed: ${imageName}`, 'success');
    await loadImages();
  } catch (e) {
    toast('Error removing image: ' + e.message, 'error');
  }
}

const btnSaveImageSettings = document.getElementById('btn-save-image-settings');
if (btnSaveImageSettings) btnSaveImageSettings.addEventListener('click', saveImageSettings);

const btnRestoreModelsJson = document.getElementById('btn-restore-models-json');
const fileRestoreModels = document.getElementById('f-restore-models-file');

if (btnRestoreModelsJson && fileRestoreModels) {
  btnRestoreModelsJson.addEventListener('click', () => fileRestoreModels.click());
  fileRestoreModels.addEventListener('change', async () => {
    const file = fileRestoreModels.files[0];
    if (!file) return;
    fileRestoreModels.value = '';

    let entries;
    try {
      entries = JSON.parse(await file.text());
    } catch (e) {
      toast('Invalid JSON file', 'error');
      return;
    }
    if (!Array.isArray(entries)) {
      toast('Expected a JSON array', 'error');
      return;
    }

    btnRestoreModelsJson.disabled = true;
    const originalHtml = btnRestoreModelsJson.innerHTML;
    btnRestoreModelsJson.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Restoring...';

    try {
      const res = await apiFetch('/api/restore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(entries),
      });
      const data = await res.json();
      if (!res.ok) {
        toast(`Restore failed: ${data.error}`, 'error');
        return;
      }

      const { summary, results } = data;
      toast(`Restore: ${summary.present} present, ${summary.queued} queued, ${summary.missing} no source${summary.errors ? `, ${summary.errors} errors` : ''}`, 'info');

      const container = document.getElementById('restore-results');
      const list = document.getElementById('restore-results-list');
      if (container && list) {
        const statusBadge = (r) => {
          if (r.status === 'present') return '<span class="badge badge-ok">present</span>';
          if (r.status === 'queued') return '<span class="badge badge-info">download queued</span>';
          if (r.status === 'missing') return '<span class="badge badge-warn">no source</span>';
          return `<span class="badge badge-warn">error: ${escHtml(r.error || '')}</span>`;
        };
        list.innerHTML = results.map(r => `
          <div class="dl-item">
            <div class="dl-item-top">
              <span class="dl-item-name">${escHtml(r.name)}</span>
              ${statusBadge(r)}
            </div>
          </div>
        `).join('');
        container.hidden = false;
      }
    } catch (e) {
      toast('Restore error: ' + e.message, 'error');
    } finally {
      btnRestoreModelsJson.disabled = false;
      btnRestoreModelsJson.innerHTML = originalHtml;
    }
  });
}

const btnPullByName = document.getElementById('btn-pull-by-name');
if (btnPullByName) btnPullByName.addEventListener('click', () => {
  const input = document.getElementById('f-image-pull-name');
  const name = input ? input.value.trim() : '';
  if (!name) { toast('Enter an image name first', 'error'); return; }
  triggerImagePull(name);
  if (input) input.value = '';
});
