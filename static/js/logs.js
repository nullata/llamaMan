// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Log Modal (shared by instances and downloads)
//   - instances: SSE live log stream + auto-scroll
//   - downloads: progress bars (overall + one per part) + collapsible raw log
// -------------------------------------------------------------------------
let logModalInstanceId = null;
let logRefreshTimer = null;
let logFetchUrl = null;       // current URL used by the log modal
let logEventSource = null;    // SSE EventSource for live log streaming
let progressTimer = null;     // download progress poll interval

function shouldAutoScroll() {
  const cb = document.getElementById('log-autoscroll');
  return cb ? cb.checked : false;
}

function startLogStream(streamUrl, fallbackUrl) {
  const out = document.getElementById('log-output');
  out.textContent = '';

  try {
    logEventSource = new EventSource(streamUrl);
    logEventSource.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.lines) {
          out.textContent += data.lines.join('');
          if (shouldAutoScroll()) out.parentElement.scrollTop = out.parentElement.scrollHeight;
        }
      } catch (e) { /* ignore parse errors */ }
    };
    logEventSource.onerror = () => {
      if (logEventSource) { logEventSource.close(); logEventSource = null; }
      logFetchUrl = fallbackUrl;
      fetchLogsPoll();
      logRefreshTimer = setInterval(fetchLogsPoll, 3000);
    };
  } catch (e) {
    logFetchUrl = fallbackUrl;
    fetchLogsPoll();
    logRefreshTimer = setInterval(fetchLogsPoll, 3000);
  }
}

async function fetchLogsPoll() {
  if (!logFetchUrl) return;
  try {
    const res = await apiFetch(logFetchUrl);
    const data = await res.json();
    const out = document.getElementById('log-output');
    if (data.lines) {
      out.textContent = data.lines.join('');
      if (shouldAutoScroll()) out.parentElement.scrollTop = out.parentElement.scrollHeight;
    } else {
      out.textContent = data.error || '(empty)';
    }
  } catch (e) {
    document.getElementById('log-output').textContent = 'Error fetching logs';
  }
}

// ---- download progress view ----

function fmtBytes(n) {
  if (n == null || isNaN(n)) return '?';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  n = Number(n);
  while (Math.abs(n) >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function progressBarHtml(pct, { stalled = false, indeterminate = false } = {}) {
  const cls = ['dlp-bar'];
  if (stalled) cls.push('dlp-bar-stalled');
  const fillCls = ['dlp-bar-fill'];
  if (indeterminate) fillCls.push('dlp-bar-indeterminate');
  const width = indeterminate ? 100 : Math.max(0, Math.min(100, pct || 0));
  return `<div class="${cls.join(' ')}"><div class="${fillCls.join(' ')}" style="width:${width}%"></div></div>`;
}

function renderDownloadProgress(data) {
  const box = document.getElementById('log-progress');
  if (!box) return;
  const dlStatus = (data && data.download_status) || 'downloading';
  const parts = (data && Array.isArray(data.parts)) ? data.parts : [];

  if (parts.length === 0) {
    const msg = dlStatus === 'downloading' ? 'Starting…' : escHtml(dlStatus);
    box.innerHTML = `<div class="dlp-empty">${msg}</div>`;
    return;
  }

  const done = dlStatus === 'completed';
  const stalled = ['paused', 'failed', 'cancelled'].includes(dlStatus);

  let totalSize = 0, totalDone = 0, sizesKnown = true;
  parts.forEach(p => {
    const partDone = (done || p.status === 'done') ? (p.size || p.downloaded || 0) : (p.downloaded || 0);
    if (p.size == null) sizesKnown = false; else totalSize += p.size;
    totalDone += partDone;
  });
  const overallPct = done ? 100
    : (sizesKnown && totalSize > 0 ? (totalDone * 100 / totalSize) : null);

  let html = '';

  if (parts.length > 1) {
    const stat = done ? 'Complete'
      : sizesKnown ? `${fmtBytes(totalDone)} / ${fmtBytes(totalSize)}${overallPct != null ? ` · ${overallPct.toFixed(0)}%` : ''}`
      : `${fmtBytes(totalDone)} downloaded`;
    html += `<div class="dlp-overall">
      <div class="dlp-head">
        <span class="dlp-title">${parts.length} files${stalled ? ` · ${escHtml(dlStatus)}` : ''}</span>
        <span class="dlp-stat">${stat}</span>
      </div>
      ${progressBarHtml(overallPct != null ? overallPct : (done ? 100 : 0), { stalled })}
    </div>`;
  }

  html += parts.map(p => {
    const partDone = p.status === 'done' || done;
    const pSize = p.size;
    const pBytes = partDone ? (pSize || p.downloaded || 0) : (p.downloaded || 0);
    const active = p.status === 'downloading' && !stalled && !partDone;
    const pct = partDone ? 100 : (pSize > 0 ? (pBytes * 100 / pSize) : (active ? null : 0));
    const speed = active && p.speed ? ` · ${fmtBytes(p.speed)}/s` : '';
    const statusCls = partDone ? 'dlp-part-done' : active ? 'dlp-part-active' : 'dlp-part-pending';
    const name = p.name ? p.name.split('/').pop() : `part ${p.index || ''}`;
    const stat = partDone ? 'done'
      : `${fmtBytes(pBytes)}${pSize != null ? ` / ${fmtBytes(pSize)}` : ''}${pct != null ? ` · ${pct.toFixed(0)}%` : ''}${speed}`;
    return `<div class="dlp-part ${statusCls}">
      <div class="dlp-head">
        <span class="dlp-name" title="${escHtml(p.name || '')}">${escHtml(name)}</span>
        <span class="dlp-stat">${stat}</span>
      </div>
      ${progressBarHtml(pct, { stalled, indeterminate: pct == null })}
    </div>`;
  }).join('');

  if (data && data.error) {
    html += `<div class="dlp-error">${escHtml(String(data.error))}</div>`;
  }
  box.innerHTML = html;
}

async function pollDownloadProgress(id, nodeId) {
  try {
    const path = `/api/downloads/${id}/progress`;
    const res = await ((typeof nodeFetch === 'function') ? nodeFetch(nodeId, path) : apiFetch(path));
    if (!res || !res.ok) return;
    const data = await res.json();
    renderDownloadProgress(data);
    if (['completed', 'failed', 'cancelled'].includes(data.download_status) && progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
  } catch (e) { /* ignore */ }
}

// ---- modal open/close ----

function openLogModal(type, id, nodeId) {
  const isDownload = type === 'download';
  const endpoint = isDownload ? 'downloads' : 'instances';
  // For a peer node, route through the cluster proxy. nodeId is empty/self for
  // this node, so the base is '' and behaviour is unchanged on single-node.
  const proxyBase = (typeof clusterProxyBase === 'function') ? clusterProxyBase(nodeId) : '';

  let title;
  if (type === 'instance') {
    const inst = instances[id];
    title = inst ? `Logs - ${inst.model_name} :${inst.port}` : 'Logs';
  } else {
    const dl = downloads[id];
    title = dl ? `Download - ${dl.repo_id}` : 'Download';
  }
  document.getElementById('log-modal-title').textContent = title;

  const progressBox = document.getElementById('log-progress');
  const out = document.getElementById('log-output');
  const autoscrollLabel = document.getElementById('log-autoscroll-label');
  const rawLogBtn = document.getElementById('btn-toggle-raw-log');

  out.textContent = 'Loading…';

  if (isDownload) {
    progressBox.hidden = false;
    progressBox.innerHTML = '<div class="dlp-empty">Loading…</div>';
    out.hidden = true;
    if (autoscrollLabel) autoscrollLabel.hidden = true;
    if (rawLogBtn) {
      rawLogBtn.hidden = false;
      rawLogBtn.classList.remove('active');
      rawLogBtn.onclick = () => {
        out.hidden = !out.hidden;
        rawLogBtn.classList.toggle('active', !out.hidden);
      };
    }
    pollDownloadProgress(id, nodeId);
    progressTimer = setInterval(() => pollDownloadProgress(id, nodeId), 1000);
  } else {
    progressBox.hidden = true;
    progressBox.innerHTML = '';
    out.hidden = false;
    if (autoscrollLabel) autoscrollLabel.hidden = false;
    if (rawLogBtn) { rawLogBtn.hidden = true; rawLogBtn.onclick = null; }
  }

  document.getElementById('log-modal').classList.add('open');
  logModalInstanceId = id;
  if (proxyBase) {
    // The cluster proxy buffers the whole response, so an SSE stream would never
    // flush. Poll the peer's log endpoint through the proxy instead.
    logFetchUrl = `${proxyBase}/api/${endpoint}/${id}/logs`;
    fetchLogsPoll();
    logRefreshTimer = setInterval(fetchLogsPoll, 3000);
  } else {
    startLogStream(`/api/${endpoint}/${id}/logs/stream`, `/api/${endpoint}/${id}/logs`);
  }
}

function closeLogs() {
  document.getElementById('log-modal').classList.remove('open');
  logModalInstanceId = null;
  logFetchUrl = null;
  if (logEventSource) { logEventSource.close(); logEventSource = null; }
  if (logRefreshTimer) { clearInterval(logRefreshTimer); logRefreshTimer = null; }
  if (progressTimer) { clearInterval(progressTimer); progressTimer = null; }
}

const closeLogsBtn = document.getElementById('btn-close-logs');
const logModalEl = document.getElementById('log-modal');
if (closeLogsBtn && logModalEl) {
  closeLogsBtn.addEventListener('click', closeLogs);
  logModalEl.addEventListener('click', (e) => {
    if (e.target === logModalEl) closeLogs();
  });
}
