// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Logging page - request-log accumulation: summary tiles, a conversations
// list, and a per-conversation drill-down (prompt/response first, metrics
// tucked into a collapsible).
// -------------------------------------------------------------------------

let _windowHours = 0;   // 0 = all time
let _refreshTimer = null;

// ---- formatting ----
function fmtInt(n) { return (n == null) ? '–' : Number(n).toLocaleString(); }

function fmtTokens(n) {
  if (n == null) return '–';
  n = Number(n);
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'k';
  return String(n);
}

function fmtTps(n) { return (n == null) ? '–' : `${Number(n).toFixed(1)} t/s`; }

function fmtMs(n) {
  if (n == null) return '–';
  return n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${Math.round(n)} ms`;
}

function fmtWhen(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '';
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return `${Math.max(0, Math.round(diff))}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return d.toLocaleString();
}

function esc(s) { return escHtml(String(s == null ? '' : s)); }

// ---- summary tiles ----
function statTile(label, value, sub) {
  return `<div class="stat-tile">
    <div class="stat-value">${value}</div>
    <div class="stat-label">${esc(label)}</div>
    ${sub ? `<div class="stat-sub">${sub}</div>` : ''}
  </div>`;
}

async function loadStats() {
  const banner = document.getElementById('logging-recording-banner');
  const grid = document.getElementById('logging-stats');
  const span = document.getElementById('logging-span');
  try {
    const q = _windowHours > 0 ? `?window_hours=${_windowHours}` : '';
    const res = await apiFetch('/api/request-log/stats' + q);
    if (!res) return;
    const d = await res.json();

    if (!d.recording) {
      banner.hidden = false;
      banner.innerHTML = '<i class="fa-solid fa-circle-info"></i> Request recording is <strong>off</strong>. '
        + 'Enable it in <a href="/">Dashboard → Settings → Application</a> to start collecting per-request stats.';
    } else {
      banner.hidden = true;
    }

    const total = (d.prompt_tokens || 0) + (d.completion_tokens || 0);
    const modeSub = d.recording_mode && d.recording_mode !== 'off'
      ? esc(d.recording_mode.replace('_', ' ')) : '';
    grid.innerHTML = [
      statTile('Requests', fmtInt(d.turn_count),
        d.error_count ? `<span class="text-danger">${fmtInt(d.error_count)} errors</span>` : modeSub),
      statTile('Avg throughput', fmtTps(d.avg_tokens_per_sec),
        d.max_tokens_per_sec != null ? `peak ${fmtTps(d.max_tokens_per_sec)}` : ''),
      statTile('Avg TTFT', fmtMs(d.avg_ttft_ms)),
      statTile('Avg latency', fmtMs(d.avg_duration_ms)),
      statTile('Completion tokens', fmtTokens(d.completion_tokens), `${fmtTokens(d.prompt_tokens)} prompt`),
      statTile('Total tokens', fmtTokens(total), d.streamed_count ? `${fmtInt(d.streamed_count)} streamed` : ''),
    ].join('');

    span.textContent = (d.first_seen_at && d.last_seen_at)
      ? `First ${new Date(d.first_seen_at).toLocaleString()} · Last ${new Date(d.last_seen_at).toLocaleString()}`
      : '';
  } catch (e) { /* ignore */ }
}

// ---- conversations ----
function withinWindow(iso) {
  if (_windowHours <= 0) return true;
  if (!iso) return false;
  const t = new Date(iso).getTime();
  if (isNaN(t)) return false;
  return (Date.now() - t) <= _windowHours * 3600 * 1000;
}

async function loadConversations() {
  const box = document.getElementById('logging-conversations');
  try {
    const res = await apiFetch('/api/request-log/conversations?limit=200');
    if (!res) return;
    let list = await res.json();
    if (!Array.isArray(list)) list = [];
    const filtered = list.filter(c => withinWindow(c.last_seen_at));

    if (filtered.length === 0) {
      box.innerHTML = `<div class="logging-empty">No recorded conversations${_windowHours > 0 ? ' in this window' : ''} yet.</div>`;
      return;
    }

    box.innerHTML = filtered.map(c => {
      const tokens = (c.prompt_tokens || 0) + (c.completion_tokens || 0);
      const title = c.title ? esc(c.title) : '<span class="logging-untitled">(no prompt text)</span>';
      const turns = Number(c.turn_count) || 0;
      return `<button type="button" class="logging-conv-row" data-id="${esc(c.conversation_id)}">
        <span class="lc-title">${title}</span>
        <span class="lc-model">${esc(c.model || '')}</span>
        <span class="lc-meta">${fmtInt(turns)} turn${turns === 1 ? '' : 's'}</span>
        <span class="lc-meta">${fmtTokens(tokens)} tok</span>
        <span class="lc-when">${esc(fmtWhen(c.last_seen_at))}</span>
      </button>`;
    }).join('');

    box.querySelectorAll('.logging-conv-row').forEach(row => {
      row.addEventListener('click', () => openConversation(row.dataset.id));
    });
  } catch (e) { /* ignore */ }
}

// ---- conversation drill-down ----
function extractPrompt(requestBody) {
  if (!requestBody) return '';
  try {
    const req = JSON.parse(requestBody);
    const msgs = req.messages;
    if (Array.isArray(msgs)) {
      for (let i = msgs.length - 1; i >= 0; i--) {
        const m = msgs[i];
        if (m && m.role === 'user' && typeof m.content === 'string') return m.content;
      }
    }
    if (typeof req.prompt === 'string') return req.prompt;
  } catch (e) { /* not JSON - fall through */ }
  return String(requestBody);
}

async function openConversation(cid) {
  const modal = document.getElementById('conv-modal');
  const body = document.getElementById('conv-body');
  document.getElementById('conv-modal-title').textContent = 'Conversation';
  body.textContent = 'Loading…';
  modal.classList.add('open');
  try {
    const res = await apiFetch(`/api/request-log/conversations/${encodeURIComponent(cid)}`);
    if (!res) return;
    if (!res.ok) { body.textContent = 'Could not load this conversation.'; return; }
    const data = await res.json();
    const turns = data.turns || [];
    document.getElementById('conv-modal-title').textContent =
      `Conversation · ${turns.length} turn${turns.length === 1 ? '' : 's'}`;

    body.innerHTML = turns.map((t, i) => {
      const prompt = extractPrompt(t.request_body);
      const response = t.response_body || '';
      const status = t.status_code;
      const statusCls = (status && status >= 400) ? 'text-danger' : 'logging-dim';
      const metrics = [
        t.endpoint ? esc(t.endpoint) : '',
        status != null ? `<span class="${statusCls}">HTTP ${esc(status)}</span>` : '',
        (t.prompt_tokens != null || t.completion_tokens != null)
          ? `${fmtInt(t.prompt_tokens)}→${fmtInt(t.completion_tokens)} tok` : '',
        t.tokens_per_sec != null ? fmtTps(t.tokens_per_sec) : '',
        t.ttft_ms != null ? `TTFT ${fmtMs(t.ttft_ms)}` : '',
        t.duration_ms != null ? `${fmtMs(t.duration_ms)} latency` : '',
        t.streamed ? 'streamed' : '',
      ].filter(Boolean).join(' · ');
      return `<div class="conv-turn">
        <div class="conv-turn-head">
          <span class="conv-turn-n">#${i + 1}</span>
          <span class="logging-dim">${esc(fmtWhen(t.created_at))}</span>
        </div>
        <div class="conv-msg"><span class="conv-msg-role">Prompt</span><pre>${esc(prompt)}</pre></div>
        <div class="conv-msg conv-msg-resp"><span class="conv-msg-role">Response</span><pre>${esc(response)}</pre></div>
        <details class="conv-metrics"><summary>Metrics</summary><div class="conv-metrics-body">${metrics || 'No metrics recorded'}</div></details>
      </div>`;
    }).join('') || '<div class="logging-empty">No turns in this conversation.</div>';
  } catch (e) {
    body.textContent = 'Error loading conversation.';
  }
}

function closeConversation() {
  document.getElementById('conv-modal').classList.remove('open');
}

// ---- wiring ----
document.getElementById('logging-window').addEventListener('click', (e) => {
  const btn = e.target.closest('.lw-btn');
  if (!btn) return;
  _windowHours = parseInt(btn.dataset.hours, 10) || 0;
  document.querySelectorAll('#logging-window .lw-btn')
    .forEach(b => b.classList.toggle('active', b === btn));
  loadStats();
  loadConversations();
});

document.getElementById('btn-close-conv').addEventListener('click', closeConversation);
document.getElementById('conv-modal').addEventListener('click', (e) => {
  if (e.target.id === 'conv-modal') closeConversation();
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeConversation(); });

// Initial load + light auto-refresh. Skip refreshing the list while a
// conversation modal is open so it doesn't yank out from under the reader.
function refreshAll() {
  loadStats();
  if (!document.getElementById('conv-modal').classList.contains('open')) {
    loadConversations();
  }
}
refreshAll();
_refreshTimer = setInterval(refreshAll, 10000);
