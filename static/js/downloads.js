// Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Download Modal
// -------------------------------------------------------------------------
async function refreshDownloadDiskSpace() {
  try {
    const node = (typeof getDownloadNode === 'function') ? getDownloadNode() : null;
    const res = await nodeFetch(node, '/api/disk-space');
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
    const res = await nodeFetch(getDownloadNode(), '/api/downloads', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (res.ok) {
      const dest = (typeof nodeNameById === 'function' && isClusterActive())
        ? ` on ${nodeNameById(getDownloadNode()) || 'node'}` : '';
      toast(`Download started: ${body.repo_id}${dest}`, 'success');
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
    const cs = window.clusterState;
    const selfName = (cs && cs.selfName) || null;
    const selfId = (cs && cs.self_id) || null;
    list.forEach(d => {
      const prev = prevDownloads[d.id];
      const item = prev ? { ...d, hasNotified: prev.hasNotified } : d;
      item._node_name = selfName; item._node_id = selfId; item._node_online = true;
      map[d.id] = item;
    });
    // In cluster mode, merge peer downloads so the list is global. They're managed
    // on their owning node, which we drive via the cluster proxy (see nodeFetch).
    if (cs && cs.enabled) {
      (cs.nodes || []).forEach(n => {
        if (n.node_id === cs.self_id) return;
        ((n.snapshot || {}).downloads || []).forEach(d => {
          map[d.id] = { ...d, _remote: true, _node_name: n.node_name, _node_id: n.node_id, _node_online: n.online };
        });
      });
    }
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

  const clusterActive = (typeof isClusterActive === 'function') && isClusterActive();
  const name = item.querySelector('.dl-item-name');
  name.textContent = label;
  name.title = dl.repo_id;
  if (clusterActive && dl._node_name) {
    const badge = document.createElement('span');
    badge.className = 'node-badge';
    badge.innerHTML = `<i class="fa-solid fa-server"></i> ${escHtml(dl._node_name)}`;
    name.appendChild(badge);
  }

  const status = item.querySelector('.dl-status');
  status.textContent = dl.status;
  status.className = `dl-status dl-status-${dl.status}`;

  const actions = item.querySelector('.dl-item-actions');
  actions.innerHTML = '';

  // A peer download is driven on its owning node via the cluster proxy. When that
  // node is offline there's no path to it, so show a read-only note instead.
  if (dl._remote && dl._node_online === false) {
    const note = document.createElement('span');
    note.className = 'meta inst-remote-note';
    note.innerHTML = `<i class="fa-solid fa-server"></i> ${escHtml(dl._node_name || 'peer')} offline`;
    actions.appendChild(note);
    return;
  }

  const nodeId = dl._node_id || '';
  const mkBtn = (cls, html) => {
    const b = document.createElement('button');
    b.className = cls;
    b.dataset.id = dl.id;
    if (nodeId) b.dataset.node = nodeId;
    b.innerHTML = html;
    actions.appendChild(b);
    return b;
  };

  mkBtn('btn-xs btn-dl-logs', '<i class="fa-solid fa-chart-line"></i> Progress');

  if (dl.status === 'downloading') {
    mkBtn('btn-xs danger btn-dl-cancel', '<i class="fa-solid fa-ban"></i> Cancel');
    mkBtn('btn-xs btn-dl-pause', '<i class="fa-solid fa-pause"></i> Pause');
  }

  if (dl.status === 'paused') {
    mkBtn('btn-xs btn-dl-resume', '<i class="fa-solid fa-play"></i> Resume');
    mkBtn('btn-xs danger btn-dl-cancel', '<i class="fa-solid fa-ban"></i> Cancel');
  }

  if (dl.status === 'completed') {
    const useBtn = mkBtn('btn-xs btn-dl-use', '<i class="fa-solid fa-arrow-right"></i> Use');
    useBtn.dataset.path = dl.dest_path;
    useBtn.dataset.filename = dl.filename || '';
  }

  if (dl.status === 'failed') {
    mkBtn('btn-xs btn-dl-retry', '<i class="fa-solid fa-rotate-right"></i> Retry');
  }

  if (['failed', 'cancelled', 'completed'].includes(dl.status)) {
    mkBtn('btn-xs danger btn-dl-remove', '<i class="fa-solid fa-trash"></i> Remove');
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

  // Bind buttons (data-node routes the call to the owning node, if remote)
  panel.querySelectorAll('.btn-dl-logs').forEach(btn => {
    btn.addEventListener('click', () => openLogModal('download', btn.dataset.id, btn.dataset.node));
  });
  panel.querySelectorAll('.btn-dl-cancel').forEach(btn => {
    btn.addEventListener('click', () => cancelDownload(btn.dataset.id, btn.dataset.node));
  });
  panel.querySelectorAll('.btn-dl-pause').forEach(btn => {
    btn.addEventListener('click', () => pauseDownload(btn.dataset.id, btn.dataset.node));
  });
  panel.querySelectorAll('.btn-dl-resume').forEach(btn => {
    btn.addEventListener('click', () => resumeDownload(btn.dataset.id, btn.dataset.node));
  });
  panel.querySelectorAll('.btn-dl-retry').forEach(btn => {
    btn.addEventListener('click', () => retryDownload(btn.dataset.id, btn.dataset.node));
  });
  panel.querySelectorAll('.btn-dl-remove').forEach(btn => {
    btn.addEventListener('click', () => removeDownload(btn.dataset.id, btn.dataset.node));
  });
  panel.querySelectorAll('.btn-dl-use').forEach(btn => {
    btn.addEventListener('click', () => {
      const fullPath = btn.dataset.filename
        ? btn.dataset.path + '/' + btn.dataset.filename
        : btn.dataset.path;
      // The model file lives on the owning node, so launch it there: carry the
      // node id so the launch form preselects it after reload (see cluster.js).
      const cs = window.clusterState || {};
      const node = btn.dataset.node;
      const nodeQ = (node && cs.self_id && node !== cs.self_id)
        ? `&launch_node=${encodeURIComponent(node)}` : '';
      window.location.href = `/?model_path=${encodeURIComponent(fullPath)}${nodeQ}`;
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

// Route to the owning node when clustering is active; a direct local call for
// self/single-node. nodeFetch / nodeSuffix live in cluster.js (always loaded).
function dlNodeFetch(nodeId, path, opts) {
  return (typeof nodeFetch === 'function') ? nodeFetch(nodeId, path, opts) : apiFetch(path, opts);
}
function dlNodeLabel(nodeId) {
  return (typeof nodeSuffix === 'function') ? nodeSuffix(nodeId) : '';
}

async function cancelDownload(id, nodeId) {
  try {
    const res = await dlNodeFetch(nodeId, `/api/downloads/${id}`, { method: 'DELETE' });
    if (res.ok) {
      toast('Download cancelled' + dlNodeLabel(nodeId), 'info');
      await pollDownloads();
    } else {
      toast('Failed to cancel download', 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function pauseDownload(id, nodeId) {
  try {
    const res = await dlNodeFetch(nodeId, `/api/downloads/${id}/pause`, { method: 'POST' });
    if (res.ok) {
      toast('Download paused' + dlNodeLabel(nodeId), 'info');
      await pollDownloads();
    } else {
      const data = await res.json();
      toast(`Failed to pause: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function resumeDownload(id, nodeId) {
  try {
    const res = await dlNodeFetch(nodeId, `/api/downloads/${id}/resume`, { method: 'POST' });
    if (res.ok) {
      toast('Download resumed' + dlNodeLabel(nodeId), 'success');
      await pollDownloads();
    } else {
      const data = await res.json();
      toast(`Failed to resume: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function retryDownload(id, nodeId) {
  try {
    const res = await dlNodeFetch(nodeId, `/api/downloads/${id}/retry`, { method: 'POST' });
    if (res.ok) {
      toast('Download retry started' + dlNodeLabel(nodeId), 'success');
      await pollDownloads();
    } else {
      const data = await res.json();
      toast(`Failed to retry: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function removeDownload(id, nodeId) {
  try {
    const res = await dlNodeFetch(nodeId, `/api/downloads/${id}/remove`, { method: 'DELETE' });
    if (res.ok) {
      toast('Download removed' + dlNodeLabel(nodeId), 'info');
      await pollDownloads();
    } else {
      const data = await res.json();
      toast(`Cannot remove: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}
