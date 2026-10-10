// Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Dashboard tiles
//
// The dashboard's stat row is derived entirely from state the existing
// pollers already maintain (instances, downloads, clusterState) plus the
// same /api/system-info and /api/gpu-info payloads the System/GPU cards
// render - no new endpoints, no extra requests. Tiles link into the view
// that owns the detail.
// -------------------------------------------------------------------------

function _tile(label, value, sub, href) {
  const inner = `<span class="tile-label">${label}</span>`
    + `<span class="tile-value">${value}</span>`
    + (sub ? `<span class="tile-sub">${sub}</span>` : '');
  return `<div class="stat-tile">${
    href ? `<a class="tile-link" href="${href}">${inner}</a>` : inner
  }</div>`;
}

function renderDashboardTiles() {
  const host = document.getElementById('stat-tiles');
  if (!host || host.offsetParent === null) return; // dashboard view hidden

  const instAll = Object.values(instances || {});
  const instActive = instAll.filter(i => i.status !== 'stopped' && i.status !== 'sleeping');
  const ready = instActive.filter(i => i.status === 'healthy').length;

  const dlAll = Object.values(downloads || {});
  const dlActive = dlAll.filter(d =>
    d.status === 'downloading' || d.status === 'moving' || d.status === 'queued');

  const cs = window.clusterState || {};
  const nodes = (cs.enabled ? (cs.nodes || []).length : 1);

  let gpuLine = '';
  let gpuValue = '-';
  const gpus = _dashGpus();
  if (gpus.length) {
    const used = gpus.reduce((s, g) => s + (g.memory_used_mb || 0), 0);
    const total = gpus.reduce((s, g) => s + (g.memory_total_mb || 0), 0);
    gpuValue = total ? `${Math.round((used / total) * 100)}%` : '-';
    gpuLine = `${gpus.length} GPU${gpus.length !== 1 ? 's' : ''}`;
  }

  host.innerHTML =
      _tile('Instances', `${ready}/${instAll.length}`, 'ready / total', '#/instances')
    + _tile('Downloads', dlActive.length, dlActive.length ? 'in progress' : 'idle', '#/downloads')
    + _tile('Models', (allModels || []).length, 'on this node', '#/models')
    + _tile('Nodes', nodes, cs.enabled ? 'cluster' : 'single node', '#/settings')
    + _tile('GPU VRAM', gpuValue, gpuLine, '#/dashboard');
}

// GPU list for the tile: local payload when clustering is off, the union of
// peer snapshots when it's on (mirrors what the dashboard's GPU card shows).
function _dashGpus() {
  const cs = window.clusterState || {};
  if (cs.enabled) {
    const out = [];
    (cs.nodes || []).forEach(n => {
      (((n.snapshot || {}).gpus) || []).forEach(g => out.push(g));
    });
    return out;
  }
  return _localGpus || [];
}

// The GPU tile reads the payload loadGpuInfo() already fetched; hook it
// without touching system.js by observing the same endpoint on the same
// cadence the card uses (10s) - cheap, and keeps the tile honest even when
// the dashboard was the first view shown.
let _localGpus = [];
async function _refreshLocalGpusForTiles() {
  if (window.__clusterActive) return;
  try {
    const res = await apiFetch('/api/gpu-info', pollOpts());
    const data = await res.json();
    _localGpus = data.gpus || [];
  } catch (e) { /* tile shows '-' */ }
}

renderDashboardTiles();
setInterval(renderDashboardTiles, 3000);
_refreshLocalGpusForTiles();
setInterval(_refreshLocalGpusForTiles, 10000);
