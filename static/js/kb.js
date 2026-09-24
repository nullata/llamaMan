// Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// MCP settings tab: expose toggle, endpoint URL, KB subsection
// -------------------------------------------------------------------------
// Self-contained: no other module reads kbState. Talks to /api/kb/* (session
// or bearer auth via the regular /api/ branch) and mirrors the MCP feature
// switches into the shared settings blob through POST /api/kb/settings.
// The KB subsection uses the launch-settings-reveal pattern (see
// toggleLaunchSectionReveal in instances.js) so the collapse animation is
// consistent with Speculative Decoding / Anti-Loop.

window.kbState = { supported: false, enabled: false, topics: [] };

let kbPollTimer = null;
let kbTopicsPage = 0;
const KB_TOPICS_PER_PAGE = 20;

async function kbApi(path, opts) {
  const res = await apiFetch(path, opts);
  if (!res) return null;
  const body = await readApiResponse(res);
  if (!res.ok) {
    toast((body && body.error) || `request failed (${res.status})`, 'error');
    return null;
  }
  return body;
}

// Client-facing port is 42069 regardless of what the operator reaches the
// UI on (5005 in local dev, whatever behind a reverse proxy). Auto-detect
// hostname so the copy-paste URL is correct for the browser's current
// origin - reverse-proxied deployments should also expose 42069 through.
function kbEndpointUrl() {
  return `${location.protocol}//${location.hostname}:42069/mcp/knowledge`;
}

// mcpServers block for Claude Desktop / Claude Code / most MCP clients. The
// Authorization header is only included when a key is needed (Per API key
// mode, or Require Authentication on) — otherwise it would be noise.
function kbNeedsKey() {
  // Per API key mode always needs a key; Global follows Require Authentication.
  return window.kbState.access === 'per_key' || !!window.kbState.require_auth;
}

function kbClientConfig() {
  const server = { type: 'http', url: kbEndpointUrl() };
  if (kbNeedsKey()) {
    server.headers = { Authorization: 'Bearer <your-api-key>' };
  }
  return JSON.stringify({ mcpServers: { 'llamaman-kb': server } }, null, 2);
}

// Clipboard API needs a secure context; plain-HTTP LAN installs fall back to
// the textarea + execCommand trick (same as the copy-command button).
async function kbCopy(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try { await navigator.clipboard.writeText(text); return true; }
    catch (_) { /* fall through */ }
  }
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
  document.body.removeChild(ta);
  return ok;
}

async function loadKbTab() {
  const status = await kbApi('/api/kb/status');
  if (!status) return;
  window.kbState = Object.assign(window.kbState, status);
  renderKbTab(status);
  if (status.supported && status.enabled) {
    await Promise.all([fetchKbTopics(), populateKbEmbeddingModels()]);
  }
}

function renderKbTab(st) {
  const banner = document.getElementById('kb-banner');
  const body = document.getElementById('kb-body');
  if (!st.supported) {
    banner.hidden = false;
    banner.textContent = st.reason === 'json_backend'
      ? 'The knowledge base needs the MariaDB backend (VECTOR search). Switch DATABASE_URL to a MariaDB 11.8+ server to use it.'
      : `The knowledge base needs MariaDB 11.8 or newer; this database reports ${st.db_version || 'an older version'}.`;
    body.hidden = true;
    return;
  }
  banner.hidden = true;
  body.hidden = false;

  document.getElementById('kb-enabled').checked = !!st.enabled;
  document.getElementById('kb-mcp-enabled').checked = !!st.mcp_enabled;
  document.getElementById('kb-mcp-access').value = st.access || 'global';
  document.getElementById('kb-mcp-ingest').checked = !!st.ingest_enabled;
  document.getElementById('kb-mcp-delete').checked = !!st.delete_enabled;
  // Prefer the new model-name setting; if empty and the legacy per-node
  // instance UUID is set, we'll leave the select empty until the dropdown
  // is populated (populateKbEmbeddingModels will preselect based on the
  // legacy instance's model_name).
  document.getElementById('kb-embedding-model').value = st.embedding_model || '';

  // Expand/collapse the KB subsection to match the stored kb_enabled state.
  // Same helper Speculative Decoding uses (instances.js).
  toggleLaunchSectionReveal(
    document.getElementById('kb-enabled-reveal'), !!st.enabled);

  const endpointRow = document.getElementById('kb-endpoint-row');
  endpointRow.hidden = !st.mcp_enabled;
  // .btn sets display, which beats the hidden attribute — toggle inline.
  document.getElementById('kb-btn-copy-config').style.display =
    st.mcp_enabled ? '' : 'none';
  if (st.mcp_enabled) {
    document.getElementById('kb-endpoint-url').textContent = kbEndpointUrl();
  }

  const meta = st.meta || {};
  const bits = [];
  if (meta.embedding_dims) {
    bits.push(meta.embedding_model
      ? `embeddings: ${meta.embedding_model} (${meta.embedding_dims}d)`
      : `embeddings: ${meta.embedding_dims}d`);
  }
  bits.push(`${st.topics || 0} topics · ${st.documents || 0} documents · ${st.chunks || 0} chunks`);
  document.getElementById('kb-meta-line').textContent = bits.join(' · ');
}

// "shared" for the pool; "key: <name>" for a key's private topic, or
// "deleted key" when the owning key no longer exists.
function kbOwnerLabel(ownerId) {
  if (!ownerId) return 'shared';
  const names = window.kbState.keyNames;
  if (!names) return 'private';
  return names[ownerId] ? `key: ${names[ownerId]}` : 'deleted key';
}

async function fetchKbTopics() {
  const topics = await kbApi('/api/kb/topics');
  if (!topics) return;
  window.kbState.topics = topics;
  // Key names for the owner label on private (per-key) topics. Best-effort:
  // without them the label falls back to "private".
  window.kbState.keyNames = null;
  try {
    const res = await apiFetch('/api/api-keys', pollOpts());
    if (res && res.ok) {
      const names = {};
      for (const k of await res.json()) names[k.id] = k.name || k.prefix || k.id;
      window.kbState.keyNames = names;
    }
  } catch (_) { /* labels only */ }
  // Filter change or delete may leave the page pointer past the end; snap back.
  if (kbTopicsPage < 0) kbTopicsPage = 0;
  renderKbTopicsList();
}

function renderKbTopicsList() {
  const list = document.getElementById('kb-topics-list');
  const pager = document.getElementById('kb-topics-pager');
  if (!list || !pager) return;
  const filter = (document.getElementById('kb-topics-filter')?.value || '')
    .toLowerCase().trim();
  const all = window.kbState.topics || [];
  const total = all.length;
  const filtered = filter
    ? all.filter(t => (t.name || '').toLowerCase().includes(filter))
    : all;

  if (filtered.length === 0) {
    list.innerHTML = `<div class="list-empty-state">${
      total === 0
        ? 'No topics yet — an MCP kb_ingest_document call creates one on demand.'
        : 'No topics match the filter.'}</div>`;
    pager.innerHTML = '';
    return;
  }

  const pages = Math.max(1, Math.ceil(filtered.length / KB_TOPICS_PER_PAGE));
  if (kbTopicsPage >= pages) kbTopicsPage = pages - 1;
  const start = kbTopicsPage * KB_TOPICS_PER_PAGE;
  const end = Math.min(start + KB_TOPICS_PER_PAGE, filtered.length);
  const page = filtered.slice(start, end);

  list.innerHTML = page.map(t => `
    <div class="dl-item" data-topic="${t.id}">
      <div class="dl-item-top">
        <span class="dl-item-name"><strong>${escHtml(t.name)}</strong>
          <span class="hint-text">${escHtml(kbOwnerLabel(t.owner_key_id))}</span></span>
        <span class="list-meta-date">${t.document_count || 0} doc(s)</span>
        <button class="btn-xs danger" data-action="kb-del-topic" data-id="${t.id}"
          title="Delete topic (removes its documents)"><i class="fa-solid fa-trash"></i> Delete</button>
      </div>
    </div>`).join('');

  const info = filter
    ? `showing ${start + 1}-${end} of ${filtered.length} filtered (${total} total)`
    : `showing ${start + 1}-${end} of ${total}`;
  const showPager = filtered.length > KB_TOPICS_PER_PAGE;
  pager.innerHTML = `<span>${info}</span>` + (showPager ? `
    <button class="btn-xs" ${kbTopicsPage === 0 ? 'disabled' : ''} data-action="kb-topics-prev">Prev</button>
    <button class="btn-xs" ${end >= filtered.length ? 'disabled' : ''} data-action="kb-topics-next">Next</button>` : '');
}

async function populateKbEmbeddingModels() {
  // Cluster-wide: dedupe embedding-marked instances by model_name so any
  // node running that model can serve embed calls. Falls back to local
  // /api/instances if the cluster endpoint is unavailable (single-node
  // installs still work identically).
  const sel = document.getElementById('kb-embedding-model');
  const seen = new Set();
  const push = (name) => {
    if (!name) return;
    seen.add(name);
  };

  try {
    const nodesRes = await apiFetch('/api/cluster/nodes', pollOpts());
    if (nodesRes && nodesRes.ok) {
      const payload = await nodesRes.json();
      const nodes = Array.isArray(payload) ? payload : (payload.nodes || []);
      for (const n of nodes) {
        const snap = n.snapshot || {};
        for (const i of (snap.instances || [])) {
          const cfg = i.config || {};
          if (cfg.embedding_model) push(i.model_name);
        }
      }
    }
  } catch (_) { /* fall through to local list */ }

  // Local /api/instances always merged in — the cluster snapshot lags one
  // heartbeat behind, so a freshly-started local embedder shows up here
  // first without waiting.
  try {
    const res = await apiFetch('/api/instances', pollOpts());
    if (res && res.ok) {
      let list = await res.json();
      if (!Array.isArray(list)) list = list.instances || [];
      for (const i of list) {
        if (i.config && i.config.embedding_model) push(i.model_name);
      }
    }
  } catch (_) { /* nothing to do */ }

  const options = [...seen].sort();
  sel.innerHTML = '<option value="">— none (start an instance with an embedding model) —</option>' +
    options.map(n => `<option value="${escHtml(n)}">${escHtml(n)}</option>`).join('');

  // Preselect: the stored model name if any, else the model_name of the
  // legacy stored instance UUID (auto-heal old configs without a save).
  const stored = window.kbState.embedding_model || '';
  if (stored && options.includes(stored)) {
    sel.value = stored;
  } else if (!stored && window.kbState.embedding_instance) {
    try {
      const res = await apiFetch('/api/instances', pollOpts());
      if (res && res.ok) {
        let list = await res.json();
        if (!Array.isArray(list)) list = list.instances || [];
        const inst = list.find(i => i.id === window.kbState.embedding_instance);
        if (inst && options.includes(inst.model_name || '')) sel.value = inst.model_name;
      }
    } catch (_) { /* nothing to do */ }
  }
}

async function kbSave(patch, reload) {
  const out = await kbApi('/api/kb/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
  if (out) {
    toast('MCP settings saved', 'success');
    if (reload) loadKbTab();
  }
}

async function kbSearch() {
  const q = document.getElementById('kb-search-query').value.trim();
  const out = document.getElementById('kb-search-results');
  if (!q) return;
  out.innerHTML = '<div class="hint-text">searching…</div>';
  const hits = await kbApi('/api/kb/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: q, limit: 6 }),
  });
  if (!hits) { out.innerHTML = ''; return; }
  if (!hits.length) { out.innerHTML = '<div class="list-empty-state">No matches.</div>'; return; }
  out.innerHTML = hits.map(h => `
    <div class="dl-item">
      <div class="dl-item-top">
        <span class="dl-item-name"><strong>${escHtml(h.title)}</strong></span>
        <span class="list-meta-date">${escHtml(h.topic)} #${h.seq} · ${(h.score * 100).toFixed(0)}%</span>
      </div>
      <div class="hint-text">${escHtml((h.text || '').slice(0, 220))}</div>
    </div>`).join('');
}

let kbReembedPolling = false;
function pollReembed() {
  if (kbReembedPolling) return;
  kbReembedPolling = true;
  const tick = async () => {
    const st = await kbApi('/api/kb/reembed/status');
    const label = document.getElementById('kb-reembed-status');
    if (!st) { kbReembedPolling = false; return; }
    if (st.running) {
      label.textContent = `re-embedding… ${st.done || 0}/${st.total || '?'} documents`;
      setTimeout(tick, 2000);
    } else {
      label.textContent = st.last_error ? `failed: ${st.last_error}` :
        (st.last_finished_at ? 'done' : '');
      kbReembedPolling = false;
      loadKbTab();
    }
  };
  tick();
}

function initKbTab() {
  const on = (id, ev, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener(ev, fn);
  };

  on('kb-enabled', 'change', e => {
    toggleLaunchSectionReveal(
      document.getElementById('kb-enabled-reveal'), e.target.checked);
    kbSave({ kb_enabled: e.target.checked }, false);
  });
  on('kb-mcp-enabled', 'change', e => kbSave({ kb_mcp_enabled: e.target.checked }, true));
  on('kb-mcp-access', 'change', e => kbSave({ kb_mcp_access: e.target.value }, true));
  on('kb-mcp-ingest', 'change', e => kbSave({ kb_mcp_ingest: e.target.checked }));
  on('kb-mcp-delete', 'change', e => kbSave({ kb_mcp_delete: e.target.checked }));
  on('kb-embedding-model', 'change', e => {
    // Save the new cluster-wide model setting AND clear the legacy per-node
    // instance UUID so the resolver's fallback path doesn't accidentally
    // fight the new selection.
    kbSave({ kb_embedding_model: e.target.value, kb_embedding_instance: '' });
  });

  // Click-to-copy the endpoint URL — small quality-of-life.
  on('kb-endpoint-url', 'click', async e => {
    const text = e.target.textContent || '';
    if (text && await kbCopy(text)) toast('Endpoint URL copied', 'success');
  });

  on('kb-btn-copy-config', 'click', async () => {
    const cfg = kbClientConfig();
    if (await kbCopy(cfg)) {
      toast(kbNeedsKey()
        ? 'MCP config copied — replace <your-api-key> with a key from the API Keys tab'
        : 'MCP config copied', 'success');
    } else {
      toast('Copy blocked; config logged to console', 'info');
      // eslint-disable-next-line no-console
      console.log(cfg);
    }
  });

  on('kb-topics-filter', 'input', () => {
    kbTopicsPage = 0;
    renderKbTopicsList();
  });

  document.getElementById('kb-topics-pager')?.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    if (btn.dataset.action === 'kb-topics-prev' && kbTopicsPage > 0) kbTopicsPage--;
    if (btn.dataset.action === 'kb-topics-next') kbTopicsPage++;
    renderKbTopicsList();
  });

  document.getElementById('kb-topics-list').addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-action="kb-del-topic"]');
    if (!btn) return;
    if (!confirm('Delete this topic and all of its documents?')) return;
    const out = await kbApi(`/api/kb/topics/${btn.dataset.id}`, { method: 'DELETE' });
    if (out) fetchKbTopics();
  });

  on('kb-btn-search', 'click', kbSearch);
  on('kb-search-query', 'keydown', e => { if (e.key === 'Enter') kbSearch(); });

  on('kb-btn-reembed', 'click', async () => {
    if (!confirm('Re-embed every document with the current embedding model? This can take a while.')) return;
    const out = await kbApi('/api/kb/reembed', { method: 'POST' });
    if (out) pollReembed();
  });

  // Refresh the tab while it is visible; cheap status call, no per-tick
  // topic/search round trips beyond what render needs.
  kbPollTimer = setInterval(() => {
    const panel = document.querySelector('[data-tab-panel="knowledge"]');
    if (panel && !panel.hidden) loadKbTab();
  }, 15000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { initKbTab(); loadKbTab(); });
} else {
  initKbTab();
  loadKbTab();
}
