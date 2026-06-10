// Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Cluster - node registry, per-node system/GPU cards, node badges
// -------------------------------------------------------------------------
// clusterState is read by instances.js (to merge remote instance cards) and by
// system.js (to yield the System/GPU cards to per-node rendering).
window.clusterState = { enabled: false, self_id: null, selfName: null, nodes: [] };

// A node silent for longer than this counts as "long offline". When the cluster
// setting "hide long-offline nodes from monitors" is on, such nodes are dropped
// from the resource-monitoring cards ONLY - they stay in the Cluster nodes list,
// the node selectors, and remain routable. This is the longer companion to the
// server-side online/offline-dot window (NODE_ONLINE_WINDOW_S).
const CLUSTER_STALE_HIDE_S = 600;  // 10 minutes

function isClusterActive() {
  return !!(window.clusterState && window.clusterState.enabled && window.clusterState.nodes.length > 0);
}

// Seconds since this node last heartbeated. Prefers the backend's skew-proof
// DB-clock age (MariaDB sets heartbeat_age_s); falls back to wall-clock from
// last_heartbeat_at (JSON backend / single host). Unknown => treat as stale.
function nodeStaleSeconds(n) {
  if (typeof n.heartbeat_age_s === 'number') return n.heartbeat_age_s;
  if (n.last_heartbeat_at) {
    const t = new Date(n.last_heartbeat_at).getTime();
    if (!isNaN(t)) return (Date.now() - t) / 1000;
  }
  return Infinity;
}

async function loadClusterNodes() {
  let data;
  try {
    const res = await apiFetch('/api/cluster/nodes');
    if (!res) return;
    data = await res.json();
  } catch (e) {
    return;
  }

  const nodes = data.nodes || [];
  const self = nodes.find(n => n.is_self);
  window.clusterState = {
    enabled: !!data.enabled,
    self_id: data.self_id || null,
    selfName: self ? self.node_name : null,
    nodes,
  };
  window.__clusterActive = isClusterActive();

  renderClusterTab(data);
  populateNodeSelects();
  if (isClusterActive()) {
    renderClusterStats(nodes);
    // Keep the library's ghost (other-node) entries fresh.
    if (typeof renderModels === 'function') renderModels();
  }
}

// -------------------------------------------------------------------------
// Node targeting - route a control call to a chosen node via the peer proxy
// -------------------------------------------------------------------------
function clusterProxyBase(nodeId) {
  const cs = window.clusterState;
  if (!cs || !cs.enabled || !nodeId || nodeId === cs.self_id) return '';
  return `/api/cluster/nodes/${nodeId}/proxy`;
}

// Like apiFetch, but targets nodeId. Local/self calls hit this node directly.
function nodeFetch(nodeId, path, opts) {
  return apiFetch(clusterProxyBase(nodeId) + path, opts);
}

function selectedNodeValue(selectId) {
  const cs = window.clusterState || {};
  if (!isClusterActive()) return cs.self_id || null;
  const sel = document.getElementById(selectId);
  return (sel && sel.value) || cs.self_id || null;
}

function getLaunchNode() { return selectedNodeValue('f-node'); }
function getImagesNode() { return selectedNodeValue('images-node'); }
function getDownloadNode() { return selectedNodeValue('d-node'); }

function nodeNameById(nodeId) {
  const cs = window.clusterState || {};
  const n = (cs.nodes || []).find(x => x.node_id === nodeId);
  return n ? n.node_name : null;
}

// " on <node>" for toasts when an action targets a peer; "" for self/single-node.
function nodeSuffix(nodeId) {
  const cs = window.clusterState || {};
  if (!isClusterActive() || !nodeId || nodeId === cs.self_id) return '';
  const name = nodeNameById(nodeId);
  return name ? ` on ${name}` : '';
}

// Fill every .node-select with the current node list; hide the selectors
// entirely when clustering is inactive so single-node UI is unchanged.
function populateNodeSelects() {
  const cs = window.clusterState || {};
  const active = isClusterActive();
  document.querySelectorAll('.node-select-row').forEach(row => { row.hidden = !active; });
  if (!active) return;
  ['f-node', 'images-node', 'd-node'].forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = '';
    (cs.nodes || []).forEach(n => {
      const opt = document.createElement('option');
      opt.value = n.node_id;
      opt.textContent = n.node_name + (n.is_self ? ' (this node)' : '') + (n.online ? '' : ' [offline]');
      sel.appendChild(opt);
    });
    if ((cs.nodes || []).some(n => n.node_id === prev)) sel.value = prev;
    else if (cs.self_id) sel.value = cs.self_id;
  });
  applyPendingLaunchNode();
}

// Cross-node "Use" on a peer's completed download reloads with ?launch_node=<id>
// so the file's owning node is preselected in the launch form. Apply it once, the
// first time the selects are populated, then let the user's choice stand.
let _appliedLaunchNodeParam = false;
function applyPendingLaunchNode() {
  if (_appliedLaunchNodeParam) return;
  const want = new URLSearchParams(location.search).get('launch_node');
  if (!want) { _appliedLaunchNodeParam = true; return; }
  const sel = document.getElementById('f-node');
  if (!sel || !sel.options.length) return;  // not populated yet; try next poll
  if ([...sel.options].some(o => o.value === want)) {
    sel.value = want;
    if (typeof onLaunchNodeChanged === 'function') onLaunchNodeChanged();
    else if (typeof updatePortSuggestion === 'function') updatePortSuggestion();
  }
  _appliedLaunchNodeParam = true;
}

// -------------------------------------------------------------------------
// Cluster settings tab
// -------------------------------------------------------------------------
function renderClusterTab(data) {
  const disabledHint = document.getElementById('cluster-disabled-hint');
  const selfCard = document.getElementById('cluster-self-card');
  const nodesLabel = document.getElementById('cluster-nodes-label');
  const list = document.getElementById('cluster-nodes-list');
  if (!list) return;

  if (!data.enabled) {
    if (disabledHint) disabledHint.hidden = false;
    if (selfCard) selfCard.hidden = true;
    if (nodesLabel) nodesLabel.hidden = true;
    list.innerHTML = '';
    return;
  }
  if (disabledHint) disabledHint.hidden = true;
  if (nodesLabel) nodesLabel.hidden = false;

  const nodes = data.nodes || [];
  const self = nodes.find(n => n.is_self);
  if (selfCard && self) {
    selfCard.hidden = false;
    selfCard.innerHTML = `
      <div class="cluster-self">
        <div class="cluster-self-row"><span class="cluster-k">This node</span>
          <strong>${escHtml(self.node_name || '')}</strong> ${nodeOnlineDot(self)}</div>
        <div class="cluster-self-row"><span class="cluster-k">Advertise URL</span>
          <code>${escHtml(self.advertise_url || '(not set)')}</code></div>
        <div class="cluster-self-row"><span class="cluster-k">Vendor / image</span>
          <span>${escHtml(self.vendor || 'cpu')} · <code>${escHtml(self.llama_image || '')}</code></span></div>
        <div class="cluster-self-row"><span class="cluster-k">Node ID</span>
          <code>${escHtml((self.node_id || '').slice(0, 12))}</code></div>
      </div>`;
  }

  list.innerHTML = nodes.map(n => {
    const snap = n.snapshot || {};
    const instCount = (snap.instances || []).filter(i => i.status !== 'stopped' && i.status !== 'sleeping').length;
    // Prefer the DB-clock age (skew-proof) over a wall-clock timestamp.
    const hb = (n.heartbeat_age_s != null)
      ? `${Math.max(0, Math.round(n.heartbeat_age_s))}s ago`
      : (n.last_heartbeat_at ? new Date(n.last_heartbeat_at).toLocaleTimeString() : '');
    const selfTag = n.is_self ? '<span class="cluster-self-tag">this node</span>' : '';
    return `
      <div class="dl-item">
        <div class="dl-item-top">
          <span class="dl-item-name">${nodeOnlineDot(n)} <strong>${escHtml(n.node_name || '')}</strong> ${selfTag}</span>
          <code class="list-meta-code" title="${escHtml(n.advertise_url || '')}">${escHtml(n.advertise_url || '(no url)')}</code>
          <span class="list-meta-date">${escHtml(n.vendor || 'cpu')}</span>
          <span class="list-meta-date">${instCount} running</span>
          <span class="list-meta-date">${n.online ? 'heartbeat' : 'last seen'} ${escHtml(hb)}</span>
          ${nodeReachBadge(n)}
        </div>
      </div>`;
  }).join('');
}

function nodeOnlineDot(n) {
  const cls = n.online ? 'cluster-dot-online' : 'cluster-dot-offline';
  const title = n.online ? 'online' : 'offline';
  return `<span class="cluster-dot ${cls}" title="${title}"></span>`;
}

// HTTP reachability of a peer FROM this node - the path dispatch/work-stealing
// actually use. A node can heartbeat into the shared DB (so it shows "online")
// while being unreachable over HTTP; that gap silently breaks load balancing,
// so call it out loudly.
function nodeReachBadge(n) {
  if (n.is_self || !n.online) return '';
  if (n.reachable) {
    return `<span class="list-meta-date text-success" title="HTTP reachable from this node">&#10003; reachable</span>`;
  }
  const why = n.reach_error ? ` (${escHtml(n.reach_error)})` : '';
  return `<span class="list-meta-date text-danger" title="Cross-node balancing & work-stealing cannot work until this is fixed${why}">&#10007; UNREACHABLE  no balancing/steal${why}</span>`;
}

// -------------------------------------------------------------------------
// Per-node System + GPU cards (reuses the single-node bar markup)
// -------------------------------------------------------------------------
function renderClusterStats(nodes) {
  // The cluster setting "hide long-offline nodes from monitors" trims nodes silent
  // past CLUSTER_STALE_HIDE_S from THESE resource cards only - they remain in the
  // Settings > Cluster nodes list, in the node selectors, and routable. Self (the
  // local node) is always shown.
  const hideStale = !!document.getElementById('s-cluster-hide-offline-monitoring')?.checked;
  const shown = hideStale
    ? nodes.filter(n => n.is_self || nodeStaleSeconds(n) <= CLUSTER_STALE_HIDE_S)
    : nodes;

  const coresLabel = document.getElementById('system-cores');
  if (coresLabel) coresLabel.textContent = `${shown.length} node${shown.length !== 1 ? 's' : ''}`;

  const sysContainer = document.getElementById('system-info-bars');
  if (sysContainer) {
    sysContainer.innerHTML = shown.map(n => `
      <div class="cluster-node-group">
        <div class="cluster-node-head">${nodeOnlineDot(n)} <strong>${escHtml(n.node_name || '')}</strong>
          ${clusterCoresLabel(n)}</div>
        ${systemBarsHtml((n.snapshot || {}).system)}
      </div>`).join('');
  }

  const gpuCard = document.getElementById('gpu-vram-card');
  const gpuContainer = document.getElementById('gpu-vram-bars');
  if (gpuCard) gpuCard.style.display = '';
  if (gpuContainer) {
    gpuContainer.innerHTML = shown.map(n => `
      <div class="cluster-node-group">
        <div class="cluster-node-head">${nodeOnlineDot(n)} <strong>${escHtml(n.node_name || '')}</strong></div>
        ${gpuBarsHtml((n.snapshot || {}).gpus)}
      </div>`).join('');
  }
}

function clusterCoresLabel(n) {
  const cores = ((n.snapshot || {}).system || {}).cpu_cores;
  return cores != null ? `<span class="text-meta">${cores} cores</span>` : '';
}

function barColor(pct) {
  return pct > 90 ? 'var(--red)' : pct > 70 ? 'var(--yellow)' : 'var(--green)';
}

function systemBarsHtml(d) {
  if (!d || d.cpu_percent == null) return '<div class="cluster-stat-empty">No stats</div>';
  const cpuPct = Math.round(d.cpu_percent);
  const ramPct = Math.round(d.ram_percent);
  const ramUsedGB = (d.ram_used_mb / 1024).toFixed(1);
  const ramTotalGB = (d.ram_total_mb / 1024).toFixed(1);
  return `
    <div class="gpu-bar-row">
      <span class="gpu-bar-label">CPU</span>
      <div class="gpu-bar-track"><div class="gpu-bar-fill" style="width:${cpuPct}%;background:${barColor(cpuPct)};"></div></div>
      <span class="gpu-bar-text">${cpuPct}%</span>
    </div>
    <div class="gpu-bar-row">
      <span class="gpu-bar-label">RAM</span>
      <div class="gpu-bar-track"><div class="gpu-bar-fill" style="width:${ramPct}%;background:${barColor(ramPct)};"></div></div>
      <span class="gpu-bar-text">${ramUsedGB} / ${ramTotalGB} GB (${ramPct}%)</span>
    </div>`;
}

function gpuBarsHtml(gpus) {
  if (!gpus || gpus.length === 0) return '<div class="cluster-stat-empty">No GPU detected</div>';
  return gpus.map(gpu => {
    const vramPct = Math.round((gpu.memory_used_mb / gpu.memory_total_mb) * 100);
    const corePct = gpu.utilization_pct ?? 0;
    const tempVal = Number.isFinite(gpu.temperature_c) ? gpu.temperature_c : null;
    const tempColor = tempVal == null ? 'var(--muted)' : tempVal >= 85 ? 'var(--red)' : tempVal >= 75 ? 'var(--yellow)' : 'var(--muted)';
    const tempHtml = tempVal == null
      ? '<span class="gpu-bar-temp" style="color:var(--muted);">-</span>'
      : `<span class="gpu-bar-temp" style="color:${tempColor};">${tempVal}&deg;C</span>`;
    return `
      <div class="gpu-bar-row">
        <div class="gpu-bar-label-col" title="${escHtml(gpu.name)}">
          <span class="gpu-bar-label">GPU ${gpu.index}</span>
          ${tempHtml}
        </div>
        <div style="flex:1;display:flex;flex-direction:column;gap:3px;">
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:0.75em;width:3em;color:var(--muted);">core</span>
            <div class="gpu-bar-track" style="flex:1;"><div class="gpu-bar-fill" style="width:${corePct}%;background:${barColor(corePct)};"></div></div>
            <span class="gpu-bar-text" style="width:3.5em;">${corePct}%</span>
          </div>
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:0.75em;width:3em;color:var(--muted);">VRAM</span>
            <div class="gpu-bar-track" style="flex:1;"><div class="gpu-bar-fill" style="width:${vramPct}%;background:${barColor(vramPct)};"></div></div>
            <span class="gpu-bar-text" style="width:3.5em;">${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB</span>
          </div>
        </div>
      </div>`;
  }).join('');
}

// Node badge for instance cards (used by instances.js when cluster is active).
function instanceNodeBadge(inst) {
  if (!isClusterActive() || !inst._node_name) return '';
  return ` <span class="node-badge"><i class="fa-solid fa-server"></i> ${escHtml(inst._node_name)}</span>`;
}

// Queue-group badge for instance cards. Shown whenever share_queue_group is set
// on an instance (independent of cluster mode - share_queue groups instances on
// the same node too), so operators can see at a glance which instances pool
// together. Empty / unset group => no badge.
function instanceQueueGroupBadge(inst) {
  const group = (inst.config && inst.config.share_queue_group) || '';
  if (!group) return '';
  return ` <span class="node-badge" title="Queue group"><i class="fa-solid fa-layer-group"></i> ${escHtml(group)}</span>`;
}
