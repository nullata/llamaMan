// Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Instance polling
// -------------------------------------------------------------------------

async function pollInstances() {
  try {
    const res = await apiFetch('/api/instances');
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
    renderInstances();
  } catch (e) { /* ignore */ }
}

async function pollContainerStats() {
  try {
    const res = await apiFetch('/api/instances/container-stats');
    const local = (res && res.ok) ? await res.json() : {};

    // Cluster: a peer's running-instance resource bars need that peer's live
    // container stats (a docker stats call), which is too heavy to ride the 5s
    // heartbeat snapshot. Pull them straight from each online peer - but on a
    // gentler cadence than the local 3s poll so we don't pile load onto peers
    // (peer load is exactly what makes them flap). Stats persist between the
    // throttled refreshes so remote bars don't blink out in between.
    const cs = window.clusterState;
    if (cs && cs.enabled && typeof nodeFetch === 'function') {
      if (_peerStatsTick++ % 3 === 0) {  // ~every 9s
        const peers = (cs.nodes || []).filter(n => n.node_id !== cs.self_id && n.online);
        const next = {};
        await Promise.all(peers.map(async (n) => {
          try {
            const r = await nodeFetch(n.node_id, '/api/instances/container-stats');
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
function updateProxySamplingOverrideState() {
  const enabled = !!document.getElementById('f-proxy-sampling-override-enabled')?.checked;
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

function updateSpecState() {
  const enabled = !!document.getElementById('f-spec-enabled')?.checked;
  ['f-spec-draft-n-max'].forEach((id) => {
    const input = document.getElementById(id);
    if (input) input.disabled = !enabled;
  });
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
    idle_timeout_min: parseInt(document.getElementById('f-idle-timeout').value) || 0,
    max_concurrent: parseInt(document.getElementById('f-max-concurrent').value) || 0,
    max_queue_depth: parseInt(document.getElementById('f-max-queue-depth').value) || 200,
    share_queue: document.getElementById('f-share-queue').checked,
    share_queue_group: document.getElementById('f-share-queue-group')?.value.trim() || '',
    share_queue_fallback: document.getElementById('f-share-queue-fallback')?.checked || false,
    auto_restart_on_crash: document.getElementById('f-auto-restart').checked,
    embedding_model: document.getElementById('f-embedding-model').checked,
    spec_enabled: document.getElementById('f-spec-enabled').checked,
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
  const threads = document.getElementById('f-threads').value.trim();
  if (threads) body.threads = parseInt(threads);
  const memoryLimit = document.getElementById('f-memory-limit').value.trim();
  if (memoryLimit) body.memory_limit = memoryLimit;
  const parallel = document.getElementById('f-parallel').value.trim();
  if (parallel) body.parallel = parseInt(parallel);
  const specNMax = document.getElementById('f-spec-draft-n-max').value.trim();
  if (specNMax) body.spec_draft_n_max = parseInt(specNMax, 10);
  const image = document.getElementById('f-image')?.value;
  if (image) body.image = image;
  return body;
}

const launchForm = document.getElementById('launch-form');
if (launchForm) launchForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn-launch');
  const status = document.getElementById('launch-status');
  btn.disabled = true;
  status.textContent = 'Launching…';

  try {
    const body = readLaunchForm();
    body.model_path = document.getElementById('f-model-path').value.trim();
    body.port = parseInt(document.getElementById('f-port').value);

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
    if (result.cancelled) {
      status.textContent = '';
      return;
    }

    const { res, data } = result;
    if (res.ok) {
      const msg = data.internal_port != null
        ? `Instance launched: public ${data.port}, llama-server ${data.internal_port}`
        : `Instance launched on port ${data.port}`;
      toast(msg, 'success');
      status.textContent = '';
      updatePortSuggestion();
      await pollInstances();
    } else {
      toast(`Launch failed: ${data.error}`, 'error');
      status.textContent = '';
    }
  } catch (e) {
    toast('Launch error: ' + e.message, 'error');
    status.textContent = '';
  } finally {
    btn.disabled = false;
  }
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
