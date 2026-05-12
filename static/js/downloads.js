// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Download Modal
// -------------------------------------------------------------------------
async function refreshDownloadDiskSpace() {
  try {
    const res = await apiFetch('/api/disk-space');
    const data = await res.json();
    if (data.free_gb != null) {
      document.getElementById('dl-disk-space').textContent =
        `${data.free_gb} GB free / ${data.total_gb} GB total`;
    }
  } catch (e) { /* ignore */ }
}

async function openDownloadModal() {
  if (typeof loadHuggingFaceTokens === 'function') {
    await loadHuggingFaceTokens();
  }
  document.getElementById('download-modal').classList.add('open');
  document.getElementById('d-repo-id').focus();
  refreshDownloadDiskSpace();
}

function closeDownloadModal() {
  document.getElementById('download-modal').classList.remove('open');
  document.getElementById('dl-form-status').textContent = '';
}

const btnOpenDownload = document.getElementById('btn-open-download');
if (btnOpenDownload) btnOpenDownload.addEventListener('click', openDownloadModal);

const btnCloseDownload = document.getElementById('btn-close-download');
if (btnCloseDownload) btnCloseDownload.addEventListener('click', closeDownloadModal);

const downloadModalEl = document.getElementById('download-modal');
if (downloadModalEl) downloadModalEl.addEventListener('click', (e) => {
  if (e.target === downloadModalEl) closeDownloadModal();
});

const downloadForm = document.getElementById('download-form');
if (downloadForm) downloadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn-start-download');
  const status = document.getElementById('dl-form-status');
  btn.disabled = true;
  status.textContent = 'Starting…';

  const body = {
    repo_id:  document.getElementById('d-repo-id').value.trim(),
    filename: document.getElementById('d-filename').value.trim(),
    hf_token_id: document.getElementById('d-token-id').value.trim(),
    speed_limit_mbps: parseFloat(document.getElementById('d-speed-limit').value) || 0,
  };

  try {
    const res = await apiFetch('/api/downloads', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (res.ok) {
      toast(`Download started: ${body.repo_id}`, 'success');
      closeDownloadModal();
      document.getElementById('download-form').reset();
      await pollDownloads();
      // Expand downloads panel if collapsed
      const panel = document.getElementById('downloads-panel');
      const hdr = document.getElementById('dl-section-toggle');
      if (panel && hdr) {
        panel.classList.remove('hidden');
        hdr.classList.remove('collapsed');
      }
    } else {
      toast(`Download failed: ${data.error}`, 'error');
      status.textContent = '';
    }
  } catch (err) {
    toast('Error: ' + err.message, 'error');
    status.textContent = '';
  } finally {
    btn.disabled = false;
  }
});

// -------------------------------------------------------------------------
// Downloads panel
// -------------------------------------------------------------------------
async function pollDownloads() {
  try {
    const res = await apiFetch('/api/downloads');
    const list = await res.json();
    const prevDownloads = downloads;
    const map = {};
    list.forEach(d => {
      const prev = prevDownloads[d.id];
      map[d.id] = prev ? { ...d, hasNotified: prev.hasNotified } : d;
    });
    downloads = map;
    renderDownloads();
  } catch (e) { /* ignore */ }
}

function createDownloadItem(dl) {
  const item = document.createElement('div');
  item.className = 'dl-item';
  item.dataset.id = dl.id;
  item.innerHTML = `
    <div class="dl-item-top">
      <span class="dl-item-name"></span>
      <span class="dl-status"></span>
    </div>
    <div class="dl-item-actions"></div>
  `;
  return item;
}

function updateDownloadItem(item, dl) {
  item.dataset.id = dl.id;

  const label = dl.filename ? dl.filename : dl.repo_id.split('/').pop();
  const top = item.querySelector('.dl-item-top');
  let spinner = top.querySelector('.spinner');
  if (dl.status === 'downloading') {
    if (!spinner) {
      spinner = document.createElement('span');
      spinner.className = 'spinner';
      top.prepend(spinner);
    }
  } else if (spinner) {
    spinner.remove();
  }

  const name = item.querySelector('.dl-item-name');
  name.textContent = label;
  name.title = dl.repo_id;

  const status = item.querySelector('.dl-status');
  status.textContent = dl.status;
  status.className = `dl-status dl-status-${dl.status}`;

  const actions = item.querySelector('.dl-item-actions');
  actions.innerHTML = '';

  const logsBtn = document.createElement('button');
  logsBtn.className = 'btn-xs btn-dl-logs';
  logsBtn.dataset.id = dl.id;
  logsBtn.innerHTML = '<i class="fa-solid fa-chart-line"></i> Progress';
  actions.appendChild(logsBtn);

  if (dl.status === 'downloading') {
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'btn-xs danger btn-dl-cancel';
    cancelBtn.dataset.id = dl.id;
    cancelBtn.innerHTML = '<i class="fa-solid fa-ban"></i> Cancel';
    actions.appendChild(cancelBtn);

    const pauseBtn = document.createElement('button');
    pauseBtn.className = 'btn-xs btn-dl-pause';
    pauseBtn.dataset.id = dl.id;
    pauseBtn.innerHTML = '<i class="fa-solid fa-pause"></i> Pause';
    actions.appendChild(pauseBtn);
  }

  if (dl.status === 'paused') {
    const resumeBtn = document.createElement('button');
    resumeBtn.className = 'btn-xs btn-dl-resume';
    resumeBtn.dataset.id = dl.id;
    resumeBtn.innerHTML = '<i class="fa-solid fa-play"></i> Resume';
    actions.appendChild(resumeBtn);

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'btn-xs danger btn-dl-cancel';
    cancelBtn.dataset.id = dl.id;
    cancelBtn.innerHTML = '<i class="fa-solid fa-ban"></i> Cancel';
    actions.appendChild(cancelBtn);
  }

  if (dl.status === 'completed') {
    const useBtn = document.createElement('button');
    useBtn.className = 'btn-xs btn-dl-use';
    useBtn.dataset.path = dl.dest_path;
    useBtn.dataset.filename = dl.filename || '';
    useBtn.innerHTML = '<i class="fa-solid fa-arrow-right"></i> Use';
    actions.appendChild(useBtn);
  }

  if (dl.status === 'failed') {
    const retryBtn = document.createElement('button');
    retryBtn.className = 'btn-xs btn-dl-retry';
    retryBtn.dataset.id = dl.id;
    retryBtn.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Retry';
    actions.appendChild(retryBtn);
  }

  if (['failed', 'cancelled', 'completed'].includes(dl.status)) {
    const removeBtn = document.createElement('button');
    removeBtn.className = 'btn-xs danger btn-dl-remove';
    removeBtn.dataset.id = dl.id;
    removeBtn.innerHTML = '<i class="fa-solid fa-trash"></i> Remove';
    actions.appendChild(removeBtn);
  }
}

function renderDownloads() {
  const panel = document.getElementById('downloads-panel');
  if (!panel) return;
  const all = Object.values(downloads);

  if (all.length === 0) {
    panel.innerHTML = '<div id="dl-empty">No downloads yet.</div>';
    return;
  }

  // Sort: active first, then by started_at desc
  all.sort((a, b) => {
    const aActive = a.status === 'downloading' ? 1 : 0;
    const bActive = b.status === 'downloading' ? 1 : 0;
    return bActive - aActive || b.started_at - a.started_at;
  });

  const emptyState = panel.querySelector('#dl-empty');
  if (emptyState) emptyState.remove();

  const existingItems = new Map(
    [...panel.querySelectorAll('.dl-item')].map(item => [item.dataset.id, item]),
  );
  let insertBeforeNode = panel.querySelector('.dl-item');

  all.forEach(dl => {
    const item = existingItems.get(String(dl.id)) || createDownloadItem(dl);
    updateDownloadItem(item, dl);
    if (item !== insertBeforeNode) {
      panel.insertBefore(item, insertBeforeNode || null);
    }
    insertBeforeNode = item.nextElementSibling;
    existingItems.delete(String(dl.id));
  });

  existingItems.forEach(item => item.remove());

  // Bind buttons
  panel.querySelectorAll('.btn-dl-logs').forEach(btn => {
    btn.addEventListener('click', () => openLogModal('download', btn.dataset.id));
  });
  panel.querySelectorAll('.btn-dl-cancel').forEach(btn => {
    btn.addEventListener('click', () => cancelDownload(btn.dataset.id));
  });
  panel.querySelectorAll('.btn-dl-pause').forEach(btn => {
    btn.addEventListener('click', () => pauseDownload(btn.dataset.id));
  });
  panel.querySelectorAll('.btn-dl-resume').forEach(btn => {
    btn.addEventListener('click', () => resumeDownload(btn.dataset.id));
  });
  panel.querySelectorAll('.btn-dl-retry').forEach(btn => {
    btn.addEventListener('click', () => retryDownload(btn.dataset.id));
  });
  panel.querySelectorAll('.btn-dl-remove').forEach(btn => {
    btn.addEventListener('click', () => removeDownload(btn.dataset.id));
  });
  panel.querySelectorAll('.btn-dl-use').forEach(btn => {
    btn.addEventListener('click', () => {
      const fullPath = btn.dataset.filename
        ? btn.dataset.path + '/' + btn.dataset.filename
        : btn.dataset.path;
      window.location.href = `/?model_path=${encodeURIComponent(fullPath)}`;
    });
  });

  // Auto-refresh models list when a download just completed
  all.forEach(dl => {
    if (dl.status === 'completed' && !dl.hasNotified) {
      dl.hasNotified = true;
      loadModels();
    }
  });
}

async function cancelDownload(id) {
  try {
    const res = await apiFetch(`/api/downloads/${id}`, { method: 'DELETE' });
    if (res.ok) {
      toast('Download cancelled', 'info');
      await pollDownloads();
    } else {
      toast('Failed to cancel download', 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function pauseDownload(id) {
  try {
    const res = await apiFetch(`/api/downloads/${id}/pause`, { method: 'POST' });
    if (res.ok) {
      toast('Download paused', 'info');
      await pollDownloads();
    } else {
      const data = await res.json();
      toast(`Failed to pause: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function resumeDownload(id) {
  try {
    const res = await apiFetch(`/api/downloads/${id}/resume`, { method: 'POST' });
    if (res.ok) {
      toast('Download resumed', 'success');
      await pollDownloads();
    } else {
      const data = await res.json();
      toast(`Failed to resume: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function retryDownload(id) {
  try {
    const res = await apiFetch(`/api/downloads/${id}/retry`, { method: 'POST' });
    if (res.ok) {
      toast('Download retry started', 'success');
      await pollDownloads();
    } else {
      const data = await res.json();
      toast(`Failed to retry: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function removeDownload(id) {
  try {
    const res = await apiFetch(`/api/downloads/${id}/remove`, { method: 'DELETE' });
    if (res.ok) {
      toast('Download removed', 'info');
      await pollDownloads();
    } else {
      const data = await res.json();
      toast(`Cannot remove: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}
