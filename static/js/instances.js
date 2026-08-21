// Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Instance polling
// -------------------------------------------------------------------------

async function pollInstances() {
  try {
    const res = await apiFetch('/api/instances', pollOpts());
    const list = await res.json();
    const map = {};
    const cs = window.clusterState;
    const selfName = (cs && cs.selfName) || null;
    const selfId = (cs && cs.self_id) || null;
    list.forEach(i => {
      i._remote = false; i._node_name = selfName; i._node_id = selfId; i._node_online = true;
      map[i.id] = i;
    });

    // In cluster mode, merge the instances published by peer nodes. Their
    // lifecycle lives on the owning node, but we drive it there via the cluster
    // proxy (see nodeFetch), so the controls below are wired to the owner.
    if (cs && cs.enabled) {
      (cs.nodes || []).forEach(n => {
        if (n.node_id === cs.self_id) return;  // self comes from the live call above
        ((n.snapshot || {}).instances || []).forEach(i => {
          map[i.id] = { ...i, _remote: true, _node_name: n.node_name, _node_id: n.node_id, _node_online: n.online };
        });
      });
    }

    instances = map;
    noteReadyTransitions(map);
    renderInstances();
  } catch (e) { /* ignore */ }
}

// -------------------------------------------------------------------------
// "Model ready" tab-title notification
// -------------------------------------------------------------------------
// Loading a model is the one long, unpredictable wait in this UI, and you spend
// it in another tab. A glyph in the title shows up in the tab strip, needs no
// permission prompt and no setting - it clears itself the moment you look at the
// page. Browsers truncate tab titles hard once you have a few open, so the glyph
// and count lead; the model name is deliberately omitted because it would be the
// first thing cut. Only fires while the tab is hidden: if you're already looking
// at the page, the card's own status badge said it first.
const _baseTitle = document.title;
let _prevInstStatus = {};    // id -> last seen status
let _readyWhileHidden = 0;
let _seeded = false;         // first poll after load only records, never announces

function updateReadyTitle() {
  document.title = _readyWhileHidden > 0
    ? `●${_readyWhileHidden > 1 ? _readyWhileHidden : ''} ${_baseTitle}`
    : _baseTitle;
}

// Two ways to qualify as "just became ready":
//   1. We watched it flip: a known id goes non-healthy -> healthy.
//   2. It showed up already healthy, but only just started.
// Case 2 exists because we frequently never see the `starting` phase at all: a
// peer's instances reach us through its heartbeat snapshot (5-15s behind) and
// browsers throttle timers in hidden tabs, which is exactly when this feature
// matters. Without it, a model launched on another node usually announces
// nothing. The freshness bound is what keeps case 2 honest - a peer returning
// from offline re-introduces all its ids as "new", but those instances started
// long ago, so they stay silent. The very first poll after a page load is
// covered by the same bound for anything older than the window, and by the
// _seeded guard for anything inside it.
const _READY_FRESH_S = 300;

function noteReadyTransitions(map) {
  const next = {};
  let becameReady = 0;
  const nowS = Date.now() / 1000;
  Object.values(map).forEach(i => {
    next[i.id] = i.status;
    const prev = _prevInstStatus[i.id];
    if (i.status !== 'healthy') return;
    if (prev) {
      if (prev !== 'healthy') becameReady++;
    } else if (_seeded && i.started_at && (nowS - i.started_at) < _READY_FRESH_S) {
      becameReady++;
    }
  });
  _prevInstStatus = next;
  _seeded = true;
  if (becameReady > 0 && document.visibilityState === 'hidden') {
    _readyWhileHidden += becameReady;
    updateReadyTitle();
  }
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    _readyWhileHidden = 0;
    updateReadyTitle();
  }
});

async function pollContainerStats() {
  try {
    const res = await apiFetch('/api/instances/container-stats', pollOpts());
    const local = (res && res.ok) ? await res.json() : {};

    // Cluster: a peer's running-instance resource bars need that peer's live
    // container stats (a docker stats call), which is too heavy to ride the 5s
    // heartbeat snapshot. Pull them straight from each online peer - but on a
    // gentler cadence than the local 3s poll so we don't pile load onto peers
    // (peer load is exactly what makes them flap). Stats persist between the
    // throttled refreshes so remote bars don't blink out in between.
    const cs = window.clusterState;
    if (cs && cs.enabled && typeof nodeFetch === 'function') {
      // Nothing to look at while the tab is hidden, so don't make peers serve
      // it. (Browsers already throttle background timers, so this mostly stops
      // the backlog that would otherwise fire the instant you tab back.) Stats
      // are left in place rather than cleared so the bars don't blink on return.
      const hidden = document.visibilityState === 'hidden';
      if (!hidden && _peerStatsTick++ % 3 === 0) {  // ~every 9s
        // `online` only means "heartbeated into the shared DB recently" - it is
        // NOT reachability, and a peer can do that while being unreachable over
        // HTTP from here (see probe_peer_reachable in api/cluster.py). Polling
        // those was self-inflicted damage: each call hung, and enough hung calls
        // exhausted the browser's per-origin connections and froze the page.
        // `reachable` is the probe the server already ran for us on this payload.
        const peers = (cs.nodes || []).filter(
          n => n.node_id !== cs.self_id && n.online && n.reachable);
        const next = {};
        await Promise.all(peers.map(async (n) => {
          try {
            const r = await nodeFetch(n.node_id, '/api/instances/container-stats',
                                      pollOpts());
            if (r && r.ok) Object.assign(next, await r.json());
          } catch (e) { /* one peer failing must not blank the others */ }
        }));
        peerContainerStats = next;
      }
    } else {
      peerContainerStats = {};
    }

    // Local wins on the (unexpected) key clash; ids are per-instance uuids.
    containerStats = { ...peerContainerStats, ...local };
    Object.entries(containerStats).forEach(([id, stat]) => {
      const el = document.querySelector(`.instance-card[data-id="${id}"] .inst-resource-line`);
      if (el) el.innerHTML = formatResourceLine(stat);
    });
  } catch (e) { /* ignore */ }
}

function formatResourceLine(stat) {
  if (!stat) return '';
  const rows = [];

  if (stat.cpu_pct != null) {
    const cores = stat.cpu_quota || stat.num_cpus || 1;
    const normalized = stat.cpu_pct / cores;
    const pct = clampPercent(normalized);
    const color = pct > 90 ? 'var(--red)' : pct > 70 ? 'var(--yellow)' : 'var(--green)';
    rows.push(`
      <div class="gpu-bar-row inst-bar-row">
        <span class="gpu-bar-label">CPU</span>
        <div class="gpu-bar-track inst-mini-bar"><div class="gpu-bar-fill" style="width:${pct}%;background:${color};"></div></div>
        <span class="gpu-bar-text">${normalized.toFixed(1)}% / ${cores} core${cores !== 1 ? 's' : ''}</span>
      </div>
    `);
  }

  if (stat.mem_used_mb != null) {
    const usedGb = (stat.mem_used_mb / 1024).toFixed(1);
    const limGb  = (stat.mem_limit_mb / 1024).toFixed(1);
    const text = stat.mem_limit_mb > 0
      ? `${usedGb} GB / ${limGb} GB`
      : `${usedGb} GB`;
    let barInner = '';
    if (stat.mem_limit_mb > 0) {
      const pct = clampPercent((stat.mem_used_mb / stat.mem_limit_mb) * 100);
      const color = pct > 90 ? 'var(--red)' : pct > 70 ? 'var(--yellow)' : 'var(--green)';
      barInner = `<div class="gpu-bar-fill" style="width:${pct}%;background:${color};"></div>`;
    }
    rows.push(`
      <div class="gpu-bar-row inst-bar-row">
        <span class="gpu-bar-label">RAM</span>
        <div class="gpu-bar-track inst-mini-bar">${barInner}</div>
        <span class="gpu-bar-text">${text}</span>
      </div>
    `);
  }

  if (stat.gpus && stat.gpus.length > 0) {
    rows.push(`<div class="inst-bar-gpu">${stat.gpus.join(', ')}</div>`);
  }

  return rows.join('');
}

function renderInstances() {
  const container = document.getElementById('instance-container');
  const count = document.getElementById('instance-count');

  const all = Object.values(instances);
  const active = all.filter(i => i.status !== 'stopped' && i.status !== 'sleeping');
  if (count) {
    count.textContent = `${active.length} instance${active.length !== 1 ? 's' : ''}`;
  }
  if (!container) return;

  // Update heading
  document.getElementById('instances-heading').textContent =
    `Running Instances (${all.length})`;

  if (all.length === 0) {
    container.innerHTML = '<div id="no-instances">No instances yet. Launch one above.</div>';
    return;
  }

  // Build ordered list: active first, then sleeping, then stopped
  const sleeping = all.filter(i => i.status === 'sleeping');
  const stopped = all.filter(i => i.status === 'stopped');
  const ordered = [...active, ...sleeping, ...stopped];

  // Preserve existing cards or build fresh
  const existingIds = new Set([...container.querySelectorAll('.instance-card')].map(el => el.dataset.id));
  const newIds = new Set(ordered.map(i => i.id));

  // Remove cards no longer present
  existingIds.forEach(id => {
    if (!newIds.has(id)) container.querySelector(`[data-id="${id}"]`)?.remove();
  });

  ordered.forEach((inst, idx) => {
    let card = container.querySelector(`[data-id="${inst.id}"]`);
    if (!card) {
      card = document.createElement('div');
      card.className = 'instance-card';
      card.dataset.id = inst.id;
      container.appendChild(card);
    }

    const uptime = (inst.status === 'stopped' || inst.status === 'sleeping')
      ? (inst.status === 'sleeping' ? 'Sleeping' : 'Down') : `Up ${formatUptime(inst.started_at)}`;
    const statusClass = `status-${inst.status}`;

    const s = inst.stats || {};
    const statsItems = [];
    if (s.model_load_time_s != null) statsItems.push(`Load ${s.model_load_time_s}s`);
    if (s.last_tokens_per_sec != null) statsItems.push(`${s.last_tokens_per_sec} t/s`);
    if (s.last_ttft_ms != null) statsItems.push(`TTFT ${s.last_ttft_ms}ms`);
    if (s.total_requests) statsItems.push(`${s.total_requests} req`);
    if (s.crash_count) statsItems.push(`<span class="text-danger">${s.crash_count} crash${s.crash_count > 1 ? 'es' : ''}</span>`);
    if (inst.last_request_at) {
      const ago = Math.round((Date.now() / 1000) - inst.last_request_at);
      if (ago < 60) statsItems.push(`last req ${ago}s ago`);
      else if (ago < 3600) statsItems.push(`last req ${Math.round(ago/60)}m ago`);
      else statsItems.push(`last req ${Math.round(ago/3600)}h ago`);
    }
    const statsLine = statsItems.length > 0
      ? `<div class="meta inst-meta-accent">${statsItems.join(' · ')}</div>` : '';

    const resourceContent = (inst.status === 'healthy' || inst.status === 'starting')
      ? formatResourceLine(containerStats[inst.id] || null) : '';
    const resourceLine = `<div class="meta inst-resource-line">${resourceContent}</div>`;

    // Queue indicator
    const q = inst.queue;
    let queueLine = '';
    if (q) {
      const qPct = q.max_queue_depth > 0 ? Math.round((q.queued / q.max_queue_depth) * 100) : 0;
      const qColor = qPct > 80 ? 'var(--red)' : qPct > 50 ? 'var(--yellow)' : 'var(--green)';
      queueLine = `<div class="meta" style="margin-top:2px;display:flex;align-items:center;gap:8px;">
        <span style="color:var(--muted);">Queue</span>
        <div style="flex:1;max-width:120px;height:8px;background:var(--surface);border-radius:3px;overflow:hidden;">
          <div style="width:${qPct}%;height:100%;background:${qColor};border-radius:3px;transition:width .3s;"></div>
        </div>
        <span style="font-size:11px;color:var(--text);font-variant-numeric:tabular-nums;">${q.active}/${q.max_concurrent} active · ${q.queued} queued</span>
      </div>`;
    }

    const portLine = inst.internal_port != null
      ? `Public ${inst.port} -> llama-server ${inst.internal_port}`
      : `Port ${inst.port}`;

    const nodeBadge = (typeof instanceNodeBadge === 'function') ? instanceNodeBadge(inst) : '';
    const queueGroupBadge = (typeof instanceQueueGroupBadge === 'function') ? instanceQueueGroupBadge(inst) : '';

    // Peer instances are managed on their owning node via the cluster proxy.
    // The data-node attribute carries that node id so each control routes there
    // (empty/self => this node, a direct local call). When the owner is offline
    // there's no path to it, so fall back to a read-only note.
    const nodeAttr = inst._node_id ? ` data-node="${escHtml(inst._node_id)}"` : '';
    const offlineRemote = inst._remote && inst._node_online === false;
    const actions = offlineRemote
      ? `<div class="inst-actions"><span class="meta inst-remote-note"><i class="fa-solid fa-server"></i> ${escHtml(inst._node_name || 'peer node')} offline</span></div>`
      : `<div class="inst-actions">
      <button class="btn btn-secondary btn-logs" data-id="${inst.id}"${nodeAttr}><i class="fa-solid fa-terminal"></i> Logs</button>
      <button class="btn btn-secondary btn-stats" data-id="${inst.id}"${nodeAttr} data-model="${escHtml(inst.model_name)}"><i class="fa-solid fa-chart-line"></i> Stats</button>
      ${inst.status !== 'stopped' && inst.status !== 'sleeping' ? `<button class="btn btn-danger btn-stop" data-id="${inst.id}"${nodeAttr}><i class="fa-solid fa-stop"></i> Stop</button>` : ''}
      ${inst.status === 'sleeping' ? `<button class="btn btn-danger btn-stop" data-id="${inst.id}"${nodeAttr}><i class="fa-solid fa-stop"></i> Stop</button>` : ''}
      ${inst.status === 'stopped' || inst.status === 'sleeping' ? `<button class="btn btn-primary btn-restart" data-id="${inst.id}"${nodeAttr}><i class="fa-solid fa-rotate-right"></i> Restart</button>` : ''}
      ${inst.status === 'stopped' ? `<button class="btn btn-danger btn-remove" data-id="${inst.id}"${nodeAttr} title="Remove from list"><i class="fa-solid fa-trash"></i></button>` : ''}
    </div>`;

    card.classList.toggle('instance-card-remote', !!inst._remote);
    card.innerHTML = `
    <div class="inst-info">
      <div class="model">${escHtml(inst.model_name)}${nodeBadge}${queueGroupBadge}</div>
      <div class="meta">${portLine} &nbsp;·&nbsp; Container ${inst.container_id ? escHtml(inst.container_id.slice(0, 12)) : '-'} &nbsp;·&nbsp; ${uptime}</div>
      ${statsLine}
      ${resourceLine}
      ${queueLine}
    </div>
    <span class="status-badge ${statusClass}">${inst.status}</span>
    ${actions}
  `;
  });

  // Move cards to correct order in DOM
  ordered.forEach(inst => {
    const card = container.querySelector(`[data-id="${inst.id}"]`);
    if (card) container.appendChild(card);
  });

  // Remove stale no-instances placeholder
  container.querySelector('#no-instances')?.remove();

  // Bind buttons (data-node routes the call to the owning node, if remote)
  container.querySelectorAll('.btn-stop').forEach(btn => {
    btn.addEventListener('click', () => stopInstance(btn.dataset.id, btn.dataset.node));
  });
  container.querySelectorAll('.btn-remove').forEach(btn => {
    btn.addEventListener('click', () => removeInstance(btn.dataset.id, btn.dataset.node));
  });
  container.querySelectorAll('.btn-restart').forEach(btn => {
    btn.addEventListener('click', () => restartInstance(btn.dataset.id, btn.dataset.node));
  });
  container.querySelectorAll('.btn-logs').forEach(btn => {
    btn.addEventListener('click', () => openLogModal('instance', btn.dataset.id, btn.dataset.node));
  });
  container.querySelectorAll('.btn-stats').forEach(btn => {
    btn.addEventListener('click', () => openStatsModal(btn.dataset.id, btn.dataset.model, btn.dataset.node));
  });
}

// -------------------------------------------------------------------------
// Instance actions
// -------------------------------------------------------------------------
function nodeFetchOr(nodeId, path, opts) {
  // Route to the owning node when clustering is active; a no-op (direct local
  // call) for self/single-node. nodeFetch lives in cluster.js (always loaded).
  return (typeof nodeFetch === 'function') ? nodeFetch(nodeId, path, opts) : apiFetch(path, opts);
}

function nodeLabel(nodeId) {
  return (typeof nodeSuffix === 'function') ? nodeSuffix(nodeId) : '';
}

async function stopInstance(id, nodeId) {
  try {
    const res = await nodeFetchOr(nodeId, `/api/instances/${id}`, { method: 'DELETE' });
    if (res.ok) {
      toast('Instance stopped' + nodeLabel(nodeId), 'info');
      await pollInstances();
      await updatePortSuggestion();
    } else {
      toast('Failed to stop instance', 'error');
    }
  } catch (e) {
    toast('Error stopping instance: ' + e.message, 'error');
  }
}

async function restartInstance(id, nodeId) {
  try {
    const attemptRestart = async (confirmOvercommit = false) => {
      const res = await nodeFetchOr(nodeId, `/api/instances/${id}/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(confirmOvercommit ? { confirm_overcommit: true } : {}),
      });
      const data = await readApiResponse(res);
      if (!res.ok && data.confirm_required) {
        const ok = await showConfirm('Launch Beyond Limit', data.error);
        if (!ok) return { cancelled: true };
        return await attemptRestart(true);
      }
      return { res, data };
    };

    const result = await attemptRestart();
    if (result.cancelled) return;

    const { res, data } = result;
    if (res.ok) {
      const msg = data.internal_port != null
        ? `Instance restarted: public ${data.port}, llama-server ${data.internal_port}`
        : `Instance restarted on port ${data.port}`;
      toast(msg + nodeLabel(nodeId), 'success');
      await pollInstances();
      await updatePortSuggestion();
    } else {
      toast(`Restart failed: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error restarting: ' + e.message, 'error');
  }
}

async function removeInstance(id, nodeId) {
  try {
    const res = await nodeFetchOr(nodeId, `/api/instances/${id}/remove`, { method: 'DELETE' });
    if (res.ok) {
      toast('Instance removed' + nodeLabel(nodeId), 'info');
      await pollInstances();
    } else {
      const data = await res.json();
      toast(`Cannot remove: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error removing instance: ' + e.message, 'error');
  }
}

// -------------------------------------------------------------------------
// Launch form
// -------------------------------------------------------------------------
// Expand/collapse a launch section's body. The inner element clips its overflow
// while the height animates (so it doesn't spill over the fields below); once
// fully open we switch to overflow:visible so upward info-tip tooltips on the
// first field row aren't cut off at the section's top edge.
function toggleLaunchSectionReveal(reveal, open) {
  if (!reveal) return;
  if (!open) {
    reveal.classList.remove('reveal-expanded');
    reveal.classList.remove('open');
    return;
  }
  reveal.classList.add('open');
  if (reveal.classList.contains('reveal-expanded')) return;
  // No transition to wait on (unsupported/disabled) — unclip immediately.
  if (parseFloat(getComputedStyle(reveal).transitionDuration) === 0) {
    reveal.classList.add('reveal-expanded');
    return;
  }
  const onEnd = (e) => {
    if (e.target !== reveal || e.propertyName !== 'grid-template-rows') return;
    reveal.removeEventListener('transitionend', onEnd);
    if (reveal.classList.contains('open')) reveal.classList.add('reveal-expanded');
  };
  reveal.addEventListener('transitionend', onEnd);
}

function updateProxySamplingOverrideState() {
  const enabled = !!document.getElementById('f-proxy-sampling-override-enabled')?.checked;
  toggleLaunchSectionReveal(document.getElementById('proxy-sampling-reveal'), enabled);
  [
    'f-proxy-sampling-temperature',
    'f-proxy-sampling-top-k',
    'f-proxy-sampling-top-p',
    'f-proxy-sampling-presence-penalty',
  ].forEach((id) => {
    const input = document.getElementById(id);
    if (input) input.disabled = !enabled;
  });
}

// --spec-type values that *require* a drafter model passed as -md. All five
// listed types accept one; only draft-mtp falls back to the main model's
// built-in MTP heads when the field is left blank, so every other draft type
// makes the drafter mandatory. Keep this in sync with the SPEC_TYPES_NEEDING_
// DRAFT_MODEL set in core/spec_decoding.py.
const SPEC_TYPES_NEEDING_DRAFT_MODEL = [
  'draft-simple', 'draft-dflash', 'draft-dspark', 'draft-eagle3',
];
const DEFAULT_SPEC_TYPE = 'draft-mtp';

function currentSpecType() {
  return document.getElementById('f-spec-type')?.value || DEFAULT_SPEC_TYPE;
}

function updateSpecState() {
  const enabled = !!document.getElementById('f-spec-enabled')?.checked;
  toggleLaunchSectionReveal(document.getElementById('spec-decoding-reveal'), enabled);
  ['f-spec-type', 'f-spec-draft-n-max', 'f-spec-draft-model'].forEach((id) => {
    const input = document.getElementById(id);
    if (input) input.disabled = !enabled;
  });
  const req = document.getElementById('spec-draft-model-req');
  if (req) req.textContent = SPEC_TYPES_NEEDING_DRAFT_MODEL.includes(currentSpecType())
    ? '(required)'
    : '(optional)';
}

function updateMmprojState() {
  const enabled = !!document.getElementById('f-mmproj-enabled')?.checked;
  toggleLaunchSectionReveal(document.getElementById('mmproj-reveal'), enabled);
  const input = document.getElementById('f-mmproj-path');
  if (input) input.disabled = !enabled;
}

// -------------------------------------------------------------------------
// GPU Settings section
// -------------------------------------------------------------------------
// Which host GPU indices this launch will make visible to llama.cpp inside
// the container. Empty gpu_devices means "all", so we default to every
// index the target node reports. Filters non-numeric junk so a stray typo
// (e.g. "0, a, 1") doesn't blow up the Auto calc.
function _visibleGpuIndicesForLaunch(allGpus) {
  const raw = (document.getElementById('f-gpu-devices')?.value || '').trim();
  if (!raw) return allGpus.map(g => g.index);
  return raw.split(',')
    .map(s => parseInt(s.trim(), 10))
    .filter(n => Number.isInteger(n) && allGpus.some(g => g.index === n));
}

// Grey out (fade + block clicks on) exactly the fields that are no-ops for
// the current launch target: Intel silently ignores per-instance GPU
// selection, GPU Layers = 0 turns off GPU entirely, and Tensor Split
// needs 2+ visible GPUs to mean anything. GPU Layers stays enabled on
// Intel because -ngl IS honored there - only the device/split knobs
// aren't. Fires on: preset load, form reset, target-node change, and
// direct edits to GPU Layers / GPU Devices.
async function updateGpuSettingsState() {
  const section = document.getElementById('gpu-settings-section');
  if (!section) return;

  const nodeId = (typeof _launchNode === 'function') ? _launchNode() : null;
  // Populate the vendor + gpu cache without blocking the first paint.
  const gpus = await fetchGpuInfoCached(nodeId);
  const vendor = (typeof cachedGpuVendor === 'function') ? cachedGpuVendor(nodeId) : null;

  const layersRaw = document.getElementById('f-gpu-layers')?.value;
  const layers = parseInt(layersRaw, 10);
  const cpuOnly = layers === 0;
  const isIntel = vendor === 'intel';

  const devicesField  = document.getElementById('f-gpu-devices')?.closest('.form-group');
  const devicesInput  = document.getElementById('f-gpu-devices');
  const splitModeField = document.getElementById('f-split-mode')?.closest('.form-group');
  const tensorGroup   = document.getElementById('f-tensor-split-group');
  const hint          = document.getElementById('gpu-settings-hint');
  const smInput       = document.getElementById('f-split-mode');
  const tsInput       = document.getElementById('f-tensor-split');

  // Device-visibility fields: greyed on Intel (silently ignored in-container)
  // or CPU-only (no GPU at all). GPU Devices is only affected here - Split
  // Mode / Tensor Split get an additional gate below.
  const devicesDisabled = isIntel || cpuOnly;
  if (devicesField) devicesField.classList.toggle('gpu-field-disabled', devicesDisabled);
  // The .gpu-field-disabled class is visual only (opacity) - we don't kill
  // pointer-events on it because that would also block tooltip hover on the
  // field's info-tip. So we have to disable the input explicitly to actually
  // block interaction.
  if (devicesInput) devicesInput.disabled = devicesDisabled;

  // Split Mode + Tensor Split need 2+ visible GPUs to matter. With one
  // visible GPU llama.cpp runs on that single device regardless of the
  // flag, so leaving the controls enabled would contradict the header
  // hint that already says "split mode has no effect".
  const visibleCount = _visibleGpuIndicesForLaunch(gpus).length;
  const splitMeaningful = !devicesDisabled && visibleCount >= 2;
  if (smInput) smInput.disabled = !splitMeaningful;
  if (splitModeField) splitModeField.classList.toggle('gpu-field-disabled', !splitMeaningful);

  // Tensor Split has one more gate: llama.cpp ignores --tensor-split when
  // --split-mode is `none`, so leaving the field editable there would let
  // users type a value that has no effect on the launch.
  const currentSplitMode = smInput?.value || 'layer';
  const tensorMeaningful = splitMeaningful && currentSplitMode !== 'none';
  if (tsInput) tsInput.disabled = !tensorMeaningful;
  if (tensorGroup) tensorGroup.classList.toggle('gpu-field-disabled', !tensorMeaningful);

  // Header hint: honest one-liner about why the section is greyed. Empty when
  // everything is in play so we don't add visual noise for the common case.
  if (hint) {
    if (isIntel) hint.textContent = 'Per-instance GPU selection is not supported on Intel.';
    else if (cpuOnly) hint.textContent = 'CPU-only (GPU Layers = 0) — no GPU placement to configure.';
    else if (visibleCount < 2) hint.textContent = visibleCount === 1
      ? 'Single visible GPU - split mode has no effect.'
      : 'No GPUs detected on the target node.';
    else if (currentSplitMode === 'none') hint.textContent = 'Split Mode = None uses a single GPU; Tensor Split is ignored.';
    else hint.textContent = '';
  }

  // Refresh the "empty = auto" preview last, once every disabled flag above
  // is in its final state (refreshTensorSplitAutoPreview keys off tsInput.disabled).
  refreshTensorSplitAutoPreview();
}

// Compute an auto tensor-split vector from the visible GPUs' total VRAM
// (24 GB + 16 GB pair -> "24,16"). Uses total (not free) so the value is
// stable across relaunches - free fluctuates with other workloads on the
// host. Rounds to whole GiB; llama.cpp normalizes the vector so no decimals
// needed. Returns null when the split is not meaningful yet (fewer than 2
// visible GPUs, or missing VRAM data) so callers can decide what to do.
async function computeAutoTensorSplit() {
  const nodeId = (typeof _launchNode === 'function') ? _launchNode() : null;
  const gpus = await fetchGpuInfoCached(nodeId);
  const visibleIdxs = _visibleGpuIndicesForLaunch(gpus);
  const visible = gpus
    .filter(g => visibleIdxs.includes(g.index))
    .sort((a, b) => a.index - b.index);
  if (visible.length < 2) return null;
  if (visible.some(g => !g.memory_total_mb)) return null;
  const weights = visible.map(g => Math.max(1, Math.round(g.memory_total_mb / 1024)));
  return {
    value: weights.join(','),
    weights,
    gpus: visible,
  };
}

// Live preview of what an empty Tensor Split will resolve to at launch. The
// hint updates whenever GPU Devices / Split Mode / GPU Layers change so the
// user always sees what "empty = auto" is going to send.
async function refreshTensorSplitAutoPreview() {
  const hint  = document.getElementById('f-tensor-split-hint');
  const input = document.getElementById('f-tensor-split');
  if (!hint || !input) return;
  // A typed value speaks for itself - don't clutter the field with a hint
  // that describes something the user has already overridden.
  if (input.value.trim() !== '') { hint.textContent = ''; return; }
  // The updater already greyed the field for these cases (Intel / CPU-only /
  // single-GPU / split-mode-none), so a preview there would be misleading.
  if (input.disabled) { hint.textContent = ''; return; }

  const auto = await computeAutoTensorSplit();
  if (!auto) {
    hint.textContent = 'Need at least 2 visible GPUs with readable VRAM to auto-split.';
    return;
  }
  const parts = auto.gpus.map((g, i) => `${auto.weights[i]} GB (GPU ${g.index})`);
  hint.textContent = `→ auto at launch: ${auto.value}   (${parts.join(' : ')} from total VRAM)`;
}

// -------------------------------------------------------------------------
// Model Settings section
// -------------------------------------------------------------------------
// V Cache Type must grey out unless Flash Attention is explicitly On:
// llama-server's llama-context.cpp guard throws "quantized V cache was
// requested, but this requires Flash Attention" if it sees a quantized
// type_v without flash-attn actually enabled. Auto is not a guarantee -
// llama.cpp resolves it per backend and may end up off - so we only unlock
// V quantization on the explicit 'on' value; Off and Auto both lock it.
// K cache has no such guard, so K Cache Type stays enabled regardless.
// If the V dropdown already sat on a quantized value when the user moves
// flash-attn away from On, we snap it back to f16 so the form can never
// submit a combination llama-server would refuse - matches the way the
// other section updaters clear stale state rather than trust the user to
// notice.
const _CACHE_TYPE_QUANTIZED = new Set(['q8_0', 'q5_1', 'q5_0', 'iq4_nl', 'q4_1', 'q4_0']);

function updateModelSettingsState() {
  const flashAttnEl = document.getElementById('f-flash-attn');
  const ctvEl = document.getElementById('f-cache-type-v');
  const ctvGroup = document.getElementById('f-cache-type-v-group');
  const hint = document.getElementById('model-settings-hint');
  if (!flashAttnEl || !ctvEl) return;

  const flashOn = flashAttnEl.value === 'on';
  ctvEl.disabled = !flashOn;
  if (ctvGroup) ctvGroup.classList.toggle('gpu-field-disabled', !flashOn);

  // Downgrade a stale quantized V to f16 the moment flash-attn leaves On,
  // so moving it to Auto or Off never leaves the form pointing at a launch
  // llama-server would reject. The user's original pick is not "remembered"
  // and re-applied when flash-attn goes back to On - that would be more
  // magic than the section is worth, and defaulting to f16 matches the safe
  // path.
  if (!flashOn && _CACHE_TYPE_QUANTIZED.has(ctvEl.value)) {
    ctvEl.value = 'f16';
  }

  if (hint) {
    hint.textContent = flashOn
      ? ''
      : 'V Cache Type quantization requires Flash Attention = On.';
  }
}

// The Quick Launch button lives in the Settings heading and only makes sense
// when the card is collapsed (an expanded card has its own Launch button) and a
// model is selected (its preset is already loaded into the hidden form).
function updateQuickLaunchVisibility() {
  const btn = document.getElementById('btn-quick-launch');
  if (!btn) return;
  const body = document.querySelector('.collapsible-body[data-section="settings"]');
  const collapsed = !!body && body.classList.contains('hidden');
  const hasModel = !!document.getElementById('f-model-path')?.value.trim();
  btn.hidden = !(collapsed && hasModel);
}

async function updatePortSuggestion() {
  const portField = document.getElementById('f-port');
  if (!portField) return;
  // Port pools are per-node, so ask the node we'd launch on.
  try {
    const node = (typeof getLaunchNode === 'function') ? getLaunchNode() : null;
    const res = await nodeFetch(node, '/api/next-port');
    const data = await res.json();
    portField.value = data.port || 8000;
  } catch (e) {
    portField.value = 8000;
  }
}

function readLaunchForm() {
  const ctxSizeRaw = document.getElementById('f-ctx-size').value.trim();
  if (!ctxSizeRaw) {
    throw new Error('Context size is required');
  }
  const ctxSize = parseInt(ctxSizeRaw, 10);
  if (!Number.isInteger(ctxSize) || ctxSize <= 0) {
    throw new Error('Context size must be a positive integer');
  }

  const body = {
    n_gpu_layers: parseInt(document.getElementById('f-gpu-layers').value),
    ctx_size: ctxSize,
    extra_args: document.getElementById('f-extra').value.trim(),
    gpu_devices: document.getElementById('f-gpu-devices').value.trim(),
    split_mode: document.getElementById('f-split-mode').value.trim(),
    tensor_split: document.getElementById('f-tensor-split').value.trim(),
    flash_attn: document.getElementById('f-flash-attn').value,
    cache_type_k: document.getElementById('f-cache-type-k').value.trim(),
    cache_type_v: document.getElementById('f-cache-type-v').value.trim(),
    idle_timeout_min: parseInt(document.getElementById('f-idle-timeout').value) || 0,
    max_concurrent: parseInt(document.getElementById('f-max-concurrent').value) || 0,
    max_queue_depth: parseInt(document.getElementById('f-max-queue-depth').value) || 200,
    share_queue: document.getElementById('f-share-queue').checked,
    share_queue_group: document.getElementById('f-share-queue-group')?.value.trim() || '',
    share_queue_fallback: document.getElementById('f-share-queue-fallback')?.checked || false,
    auto_restart_on_crash: document.getElementById('f-auto-restart').checked,
    embedding_model: document.getElementById('f-embedding-model').checked,
    spec_enabled: document.getElementById('f-spec-enabled').checked,
    spec_type: currentSpecType(),
    spec_draft_model: document.getElementById('f-spec-draft-model').value.trim(),
    mmproj_enabled: document.getElementById('f-mmproj-enabled').checked,
    mmproj_path: document.getElementById('f-mmproj-path').value.trim(),
    proxy_sampling_override_enabled: document.getElementById('f-proxy-sampling-override-enabled').checked,
    proxy_sampling_temperature: parseFloat(document.getElementById('f-proxy-sampling-temperature').value),
    proxy_sampling_top_k: parseInt(document.getElementById('f-proxy-sampling-top-k').value, 10),
    proxy_sampling_top_p: parseFloat(document.getElementById('f-proxy-sampling-top-p').value),
    proxy_sampling_presence_penalty: parseFloat(document.getElementById('f-proxy-sampling-presence-penalty').value),
    proxy_sampling_repeat_penalty: parseFloat(document.getElementById('f-proxy-sampling-repeat-penalty').value),
  };
  if (!Number.isFinite(body.proxy_sampling_temperature) || body.proxy_sampling_temperature < 0 || body.proxy_sampling_temperature > 2) {
    throw new Error('Proxy-side temperature must be between 0 and 2');
  }
  if (!Number.isInteger(body.proxy_sampling_top_k) || body.proxy_sampling_top_k < 0) {
    throw new Error('Proxy-side top k must be an integer >= 0');
  }
  if (!Number.isFinite(body.proxy_sampling_top_p) || body.proxy_sampling_top_p <= 0 || body.proxy_sampling_top_p > 1) {
    throw new Error('Proxy-side top p must be greater than 0 and no more than 1');
  }
  if (!Number.isFinite(body.proxy_sampling_presence_penalty) || body.proxy_sampling_presence_penalty < -2 || body.proxy_sampling_presence_penalty > 2) {
    throw new Error('Proxy-side presence penalty must be between -2 and 2');
  }
  if (!Number.isFinite(body.proxy_sampling_repeat_penalty) || body.proxy_sampling_repeat_penalty < 0 || body.proxy_sampling_repeat_penalty > 2) {
    throw new Error('Proxy-side repeat penalty must be between 0 and 2');
  }
  if (body.spec_enabled && SPEC_TYPES_NEEDING_DRAFT_MODEL.includes(body.spec_type) && !body.spec_draft_model) {
    throw new Error(`Speculative decoding with ${body.spec_type} requires a draft model`);
  }
  if (body.mmproj_enabled && !body.mmproj_path) {
    throw new Error('Image input requires an MMPROJ model path');
  }
  const threads = document.getElementById('f-threads').value.trim();
  if (threads) body.threads = parseInt(threads);
  const memoryLimit = document.getElementById('f-memory-limit').value.trim();
  if (memoryLimit) body.memory_limit = memoryLimit;
  const parallel = document.getElementById('f-parallel').value.trim();
  if (parallel) body.parallel = parseInt(parallel);
  const specNMax = document.getElementById('f-spec-draft-n-max').value.trim();
  if (specNMax) body.spec_draft_n_max = parseInt(specNMax, 10);
  // Advanced spec-decoding knobs. Empty = don't set the key at all, so the
  // server's parse_spec_config leaves it as None and build_llama_cmd omits
  // the flag - llama-server falls back to its own default. Same "empty means
  // auto" contract that Tensor Split uses.
  const specNMin = document.getElementById('f-spec-draft-n-min')?.value.trim();
  if (specNMin) body.spec_draft_n_min = parseInt(specNMin, 10);
  const specPSplit = document.getElementById('f-spec-draft-p-split')?.value.trim();
  if (specPSplit) body.spec_draft_p_split = parseFloat(specPSplit);
  const specPMin = document.getElementById('f-spec-draft-p-min')?.value.trim();
  if (specPMin) body.spec_draft_p_min = parseFloat(specPMin);
  const image = document.getElementById('f-image')?.value;
  if (image) body.image = image;
  return body;
}

// Shared by the in-card Launch button and the heading's Quick Launch button.
// Returns {invalidForm:true} when the form itself didn't pass validation, so a
// caller launching from the collapsed card can open it for the user to fix.
async function submitLaunchForm(btn, status) {
  btn.disabled = true;
  if (status) status.textContent = 'Launching…';

  try {
    let body;
    try {
      body = readLaunchForm();
    } catch (e) {
      toast('Launch error: ' + e.message, 'error');
      return { invalidForm: true };
    }
    body.model_path = document.getElementById('f-model-path').value.trim();
    body.port = parseInt(document.getElementById('f-port').value);

    // Empty Tensor Split means "auto from VRAM at launch". Only resolve here
    // (never at preset save) so a preset stays portable across nodes with
    // different GPU topologies - the auto-compute runs fresh on whichever
    // node ends up launching. Only substitute when the flag would actually
    // matter (layer/row + 2+ visible GPUs); None or single-GPU pass through
    // as empty because llama.cpp ignores --tensor-split there anyway.
    if (!body.tensor_split && (body.split_mode === 'layer' || body.split_mode === 'row')) {
      const auto = await computeAutoTensorSplit();
      if (auto) body.tensor_split = auto.value;
    }

    const attemptLaunch = async (confirmOvercommit = false) => {
      const launchBody = {
        ...body,
        ...(confirmOvercommit ? { confirm_overcommit: true } : {}),
      };
      const res = await nodeFetch(getLaunchNode(), '/api/instances', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(launchBody),
      });
      const data = await readApiResponse(res);
      if (!res.ok && data.confirm_required) {
        const ok = await showConfirm('Launch Beyond Limit', data.error);
        if (!ok) return { cancelled: true };
        return await attemptLaunch(true);
      }
      return { res, data };
    };

    const result = await attemptLaunch();
    if (result.cancelled) return {};

    const { res, data } = result;
    if (res.ok) {
      const msg = data.internal_port != null
        ? `Instance launched: public ${data.port}, llama-server ${data.internal_port}`
        : `Instance launched on port ${data.port}`;
      toast(msg, 'success');
      updatePortSuggestion();
      await pollInstances();
    } else {
      toast(`Launch failed: ${data.error}`, 'error');
    }
    return { launched: res.ok };
  } catch (e) {
    toast('Launch error: ' + e.message, 'error');
    return {};
  } finally {
    btn.disabled = false;
    if (status) status.textContent = '';
  }
}

const launchForm = document.getElementById('launch-form');
if (launchForm) launchForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  await submitLaunchForm(
    document.getElementById('btn-launch'),
    document.getElementById('launch-status'),
  );
});

const quickLaunchBtn = document.getElementById('btn-quick-launch');
if (quickLaunchBtn) quickLaunchBtn.addEventListener('click', async (e) => {
  e.stopPropagation();  // the heading it sits in toggles the section on click
  const result = await submitLaunchForm(quickLaunchBtn, null);
  // Nothing is visible to correct while the card is collapsed, so open it.
  if (result.invalidForm && typeof toggleSection === 'function') toggleSection('settings');
});

// -------------------------------------------------------------------------
// Preset save
// -------------------------------------------------------------------------
const savePresetBtn = document.getElementById('btn-save-preset');
if (savePresetBtn) savePresetBtn.addEventListener('click', async () => {
  const modelPath = document.getElementById('f-model-path').value.trim();
  if (!modelPath) {
    toast('Select a model first', 'error');
    return;
  }

  try {
    const body = readLaunchForm();
    body.note = (document.getElementById('f-note').value || '').trim();
    body.favorite = isModelFavorited(modelPath);
    // In cluster mode the hardware fields are this Target node's override.
    if (typeof isClusterActive === 'function' && isClusterActive()) {
      body.override_node_id = getLaunchNode();
    }
    const res = await apiFetch(`/api/presets${encodePathForUrl(modelPath)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (res.ok) {
      toast('Preset saved', 'success');
    } else {
      const data = await readApiResponse(res);
      toast(`Failed to save preset: ${data.error || 'unknown error'}`, 'error');
    }
  } catch (e) {
    toast('Error saving preset: ' + e.message, 'error');
  }
});

const proxySamplingOverrideToggle = document.getElementById('f-proxy-sampling-override-enabled');
if (proxySamplingOverrideToggle) {
  proxySamplingOverrideToggle.addEventListener('change', updateProxySamplingOverrideState);
  updateProxySamplingOverrideState();
}

const specToggle = document.getElementById('f-spec-enabled');
if (specToggle) {
  specToggle.addEventListener('change', updateSpecState);
  updateSpecState();
}

const mmprojToggle = document.getElementById('f-mmproj-enabled');
if (mmprojToggle) {
  mmprojToggle.addEventListener('change', updateMmprojState);
  updateMmprojState();
}

const specTypeSelect = document.getElementById('f-spec-type');
if (specTypeSelect) specTypeSelect.addEventListener('change', updateSpecState);

// GPU Settings section: react to the two inputs that gate the visible-GPU
// count (GPU Layers = 0 disables the whole placement group; GPU Devices
// narrows/widens the visible set that Tensor Split acts on) and hook the
// Auto button. cachedGpuVendor is only populated after fetchGpuInfoCached
// resolves the first time, so we call the updater once at module init to
// prime the fetch.
const gpuLayersField = document.getElementById('f-gpu-layers');
if (gpuLayersField) gpuLayersField.addEventListener('input', updateGpuSettingsState);
const gpuDevicesFieldForGating = document.getElementById('f-gpu-devices');
if (gpuDevicesFieldForGating) gpuDevicesFieldForGating.addEventListener('input', updateGpuSettingsState);
const splitModeField = document.getElementById('f-split-mode');
if (splitModeField) splitModeField.addEventListener('change', updateGpuSettingsState);
const tensorSplitField = document.getElementById('f-tensor-split');
// Typing in the field or clearing it must refresh the "empty = auto" preview
// below it - it shows only while the field is empty.
if (tensorSplitField) tensorSplitField.addEventListener('input', refreshTensorSplitAutoPreview);
if (typeof updateGpuSettingsState === 'function') updateGpuSettingsState();

// Model Settings: the flash-attn select (Auto/On/Off) gates the V Cache Type
// dropdown - only the explicit On value unlocks V quantization.
const flashAttnSelect = document.getElementById('f-flash-attn');
if (flashAttnSelect) flashAttnSelect.addEventListener('change', updateModelSettingsState);
if (typeof updateModelSettingsState === 'function') updateModelSettingsState();

// A model can also be set by typing a path, not just by clicking the library.
const quickLaunchModelField = document.getElementById('f-model-path');
if (quickLaunchModelField) quickLaunchModelField.addEventListener('input', updateQuickLaunchVisibility);
